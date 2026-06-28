// Map editor — in-place dirty-rect upload of paint bytes to the GPU texture.
//
// Stage 1 of the paint-path speedup. Instead of `Assets::get_mut` on the paint
// Image (which fires `AssetEvent::Modified` → Bevy deep-clones the whole Image on
// extract AND frees+reallocs+fully-reuploads the entire GPU texture on prepare,
// ~16 MiB per dab at 2048² + 2 forced material bind-group rebuilds), the brush now
// writes byte-identical RGBA8 into an authoritative CPU mirror, tracks the union of
// TEXELS ACTUALLY WRITTEN this dab as a dirty `URect`, and ships ONLY that packed
// sub-rectangle into the RenderApp. There a `Render`-schedule system ordered
// `.after(prepare_assets::<GpuImage>)` calls `RenderQueue::write_texture` IN PLACE
// against the persistent `GpuImage.texture` for just that sub-rectangle.
//
// Because the texture is updated in place (same wgpu texture + view), the
// `StandardMaterial` bind group stays valid — no material refresh is needed.
//
// Row alignment: `write_texture` requires `bytes_per_row` to be a multiple of
// `COPY_BYTES_PER_ROW_ALIGNMENT` (256) when copying more than one row. We pack the
// dirty sub-rectangle into a tightly-staged buffer whose per-row stride is the
// dirty width's byte count rounded UP to 256, so an arbitrary (non-POT) atlas edge
// and an arbitrary dirty width both upload correctly with minimal transport.

use bevy::prelude::*;
use bevy::asset::AssetId;
use bevy::image::Image;
use bevy::math::URect;
use bevy::render::{
    Render, RenderApp, RenderSystems,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::{RenderAssets, prepare_assets},
    render_resource::{
        Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    },
    renderer::RenderQueue,
    texture::GpuImage,
};


/// A pending dirty-rect upload, produced in the main world by the brush / Color All
/// and consumed once by the RenderApp. `bytes` is the packed sub-rectangle (tightly
/// packed `rect.width()*4` bytes per row, top-to-bottom over `rect`'s Y range).
///
/// Consume-once: `ExtractResource` only re-extracts on `is_changed()`, so without a
/// guard the LAST rect would re-upload every idle frame. The main-world producer
/// sets `payload` to `Some` exactly when there is new data; the RenderApp system
/// uploads it and the producer leaves it `Some` until the next write — but the
/// `generation` counter (bumped on every new payload) lets the RenderApp skip a
/// payload it has already uploaded, so idle frames (no main-world change → no
/// extract → render resource unchanged, AND a changed-but-same-generation guard)
/// never re-upload.
#[derive(Resource, Clone, Default)]
pub struct PaintDirtyRect {
    /// Texel rectangle that was actually written (atlas-texel coords, origin
    /// top-left). Empty (`payload == None`) ⇒ nothing to upload.
    pub rect: URect,
    /// Packed RGBA8 bytes of `rect`, row-major top-to-bottom, `rect.width()*4`
    /// bytes per row (tight, no padding — padding to 256 happens at upload time).
    pub payload: Option<Vec<u8>>,
    /// The paint Image asset id (to look up its `GpuImage` in the RenderApp).
    pub image_id: Option<AssetId<Image>>,
    /// Bumped on every new payload so the RenderApp uploads each at most once.
    pub generation: u64,
}

/// The render-world mirror of `PaintDirtyRect` (extracted each frame it changes).
#[derive(Resource, Default)]
pub struct RenderPaintDirtyRect {
    pub inner: PaintDirtyRect,
}

impl ExtractResource for RenderPaintDirtyRect {
    type Source = PaintDirtyRect;
    fn extract_resource(source: &Self::Source) -> Self {
        RenderPaintDirtyRect { inner: source.clone() }
    }
}


/// Plugin: registers the extract of `PaintDirtyRect` and the in-place upload system
/// in the RenderApp.
pub struct PaintUploadPlugin;

impl Plugin for PaintUploadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PaintDirtyRect>();
        app.add_plugins(ExtractResourcePlugin::<RenderPaintDirtyRect>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<LastUploadedGeneration>();
            // Ordered `.after(prepare_assets::<GpuImage>)` so the persistent
            // `GpuImage` exists before we write into it. On the first MapEditor
            // entry (texture not yet prepared) the lookup simply no-ops.
            render_app.add_systems(
                Render,
                upload_paint_dirty_rect
                    .in_set(RenderSystems::PrepareAssets)
                    .after(prepare_assets::<GpuImage>),
            );
        }
    }
}

/// The last `PaintDirtyRect::generation` the RenderApp has uploaded — the
/// consume-once guard so an unchanged (or only re-extracted) payload is not
/// re-uploaded on idle frames.
#[derive(Resource, Default)]
struct LastUploadedGeneration(u64);


/// wgpu's `COPY_BYTES_PER_ROW_ALIGNMENT` — `bytes_per_row` for a multi-row
/// `write_texture` must be a multiple of this. Fixed by the wgpu spec (defined
/// locally since `wgpu` is not a direct dependency of this crate).
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Round `bytes` up to the next multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` (256).
#[inline]
fn align_bpr(bytes: u32) -> u32 {
    let a = COPY_BYTES_PER_ROW_ALIGNMENT;
    bytes.div_ceil(a) * a
}

/// In-place upload of the packed dirty sub-rectangle into the persistent paint
/// `GpuImage.texture`. Runs in the RenderApp's `Render` schedule, `PrepareAssets`
/// set, after `prepare_assets::<GpuImage>`.
fn upload_paint_dirty_rect(
    dirty:        Res<RenderPaintDirtyRect>,
    mut last_gen:  ResMut<LastUploadedGeneration>,
    gpu_images:   Res<RenderAssets<GpuImage>>,
    queue:        Res<RenderQueue>,
) {
    let d = &dirty.inner;
    let Some(payload) = d.payload.as_ref() else { return };
    let Some(image_id) = d.image_id else { return };
    // Consume-once: skip a generation we already uploaded.
    if d.generation == last_gen.0 { return; }

    let w = d.rect.width();
    let h = d.rect.height();
    if w == 0 || h == 0 { return; }
    let tight_bpr = w * 4;
    if payload.len() < (tight_bpr as usize) * (h as usize) { return; }

    let Some(gpu_image) = gpu_images.get(image_id) else { return }; // not prepared yet
    // Defensive: never write outside the texture.
    if d.rect.max.x > gpu_image.size.width || d.rect.max.y > gpu_image.size.height {
        return;
    }

    // Stage the rows into a 256-aligned-stride buffer (write_texture needs
    // bytes_per_row aligned when copying multiple rows). One row is exempt, but a
    // single padded path covers both.
    let padded_bpr = align_bpr(tight_bpr);
    let mut staged = vec![0u8; (padded_bpr as usize) * (h as usize)];
    for row in 0..h as usize {
        let src = &payload[row * tight_bpr as usize..(row + 1) * tight_bpr as usize];
        let dst_off = row * padded_bpr as usize;
        staged[dst_off..dst_off + tight_bpr as usize].copy_from_slice(src);
    }

    queue.write_texture(
        TexelCopyTextureInfo {
            texture:   &gpu_image.texture,
            mip_level: 0,
            origin:    Origin3d { x: d.rect.min.x, y: d.rect.min.y, z: 0 },
            aspect:    TextureAspect::All,
        },
        &staged,
        TexelCopyBufferLayout {
            offset:         0,
            bytes_per_row:  Some(padded_bpr),
            rows_per_image: Some(h),
        },
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );

    last_gen.0 = d.generation;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_align_bpr_rounds_up_to_256() {
        assert_eq!(align_bpr(1), 256);
        assert_eq!(align_bpr(256), 256);
        assert_eq!(align_bpr(257), 512);
        assert_eq!(align_bpr(8192), 8192); // 2048*4, already aligned
        assert_eq!(align_bpr(0), 0);
    }
}
