// `.world` binary format — terrain mesh + painted texture + MapSize.
//
// Single source of truth for the on-disk `.world` layout. A `.world` file stores
// the WHOLE terrain (every submesh's geometry + its world `Transform`) together
// with the painted atlas texture and the world's `MapSize`, so a painted map can
// be exported from the Map Editor and reloaded as a fully-textured,
// simulation-ready world. (Organisms/colony stay in the separate `.colony` file.)
//
// Conventions mirror `colony_save_load.rs`: hand-rolled little-endian, an 8-byte
// ASCII magic with the version encoded in the trailing digits (no redundant
// `u32`), `std::io::Result` + `std::io::Error::other`, and EVERY untrusted length
// validated by `checked_count` before any `Vec::with_capacity` (a corrupt/old
// file returns a clean `Err`, never a giant alloc — see the documented 684 GB OOM
// regression).
//
// Byte layout (all little-endian):
//   magic        [u8;8] = b"AEONSW01"      (version 1 encoded in trailing digits)
//   map_x        f32
//   map_z        f32
//   tex_width    u32
//   tex_height   u32
//   tex_format   u32   (0 = Rgba8UnormSrgb; reject others)
//   tex_len      u32   (== tex_width*tex_height*4; cross-checked)
//   tex_bytes    [u8; tex_len]
//   mesh_count   u32
//   repeat mesh_count:
//     translation [f32;3]
//     rotation    [f32;4]  (quat xyzw)
//     scale       [f32;3]
//     vert_count  u32
//     positions   [f32;3] * vert_count   (submesh-LOCAL)
//     normals     [f32;3] * vert_count
//     uv0         [f32;2] * vert_count
//     index_count u32
//     indices     [u32]   * index_count

use bevy::prelude::*; // Transform, Vec3, Quat


// ── Format constants ──────────────────────────────────────────────────────────

pub const WORLD_MAGIC: &[u8; 8] = b"AEONSW01";
pub const MAX_WORLD_VERSION: u8 = 1;
/// `tex_format` tag for `Rgba8UnormSrgb` (the only format this version stores).
pub const TEX_FORMAT_RGBA8_SRGB: u32 = 0;


// ── Owned, asset-handle-free data ──────────────────────────────────────────────

/// The painted atlas texture as raw RGBA8 bytes plus its dimensions.
pub struct PaintTextureData {
    pub width:  u32,
    pub height: u32,
    pub bytes:  Vec<u8>,
}

/// One terrain submesh: its world pose plus submesh-LOCAL geometry.
pub struct WorldMeshEntry {
    pub transform: Transform,     // world pose (decomposed from GlobalTransform)
    pub positions: Vec<[f32; 3]>, // submesh-LOCAL
    pub normals:   Vec<[f32; 3]>,
    pub uv0:       Vec<[f32; 2]>,
    pub indices:   Vec<u32>,
}

/// The full decoded `.world` payload.
pub struct WorldData {
    pub map_x:   f32,
    pub map_z:   f32,
    pub texture: PaintTextureData,
    pub meshes:  Vec<WorldMeshEntry>,
}


// ── Path wrappers (what the engine calls) ───────────────────────────────────────

pub fn write_world(path: &str, data: &WorldData) -> std::io::Result<()> {
    std::fs::write(path, write_world_bytes(data))
}

pub fn read_world(path: &str) -> std::io::Result<WorldData> {
    read_world_bytes(&std::fs::read(path)?)
}


// ── Little-endian writer helpers ────────────────────────────────────────────────

#[inline]
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_vec3(out: &mut Vec<u8>, v: Vec3) {
    put_f32(out, v.x);
    put_f32(out, v.y);
    put_f32(out, v.z);
}
#[inline]
fn put_quat(out: &mut Vec<u8>, q: Quat) {
    put_f32(out, q.x);
    put_f32(out, q.y);
    put_f32(out, q.z);
    put_f32(out, q.w);
}


// ── Serialise ───────────────────────────────────────────────────────────────────

fn write_world_bytes(data: &WorldData) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(WORLD_MAGIC);
    put_f32(&mut out, data.map_x);
    put_f32(&mut out, data.map_z);

    put_u32(&mut out, data.texture.width);
    put_u32(&mut out, data.texture.height);
    put_u32(&mut out, TEX_FORMAT_RGBA8_SRGB);
    put_u32(&mut out, data.texture.bytes.len() as u32);
    out.extend_from_slice(&data.texture.bytes);

    put_u32(&mut out, data.meshes.len() as u32);
    for m in &data.meshes {
        put_vec3(&mut out, m.transform.translation);
        put_quat(&mut out, m.transform.rotation);
        put_vec3(&mut out, m.transform.scale);

        put_u32(&mut out, m.positions.len() as u32);
        for p in &m.positions {
            put_f32(&mut out, p[0]);
            put_f32(&mut out, p[1]);
            put_f32(&mut out, p[2]);
        }
        for n in &m.normals {
            put_f32(&mut out, n[0]);
            put_f32(&mut out, n[1]);
            put_f32(&mut out, n[2]);
        }
        for uv in &m.uv0 {
            put_f32(&mut out, uv[0]);
            put_f32(&mut out, uv[1]);
        }

        put_u32(&mut out, m.indices.len() as u32);
        for &i in &m.indices {
            put_u32(&mut out, i);
        }
    }
    out
}


// ── Little-endian reader helpers (cursor-style) ─────────────────────────────────

#[inline]
fn read_u32(buf: &[u8], c: &mut usize) -> std::io::Result<u32> {
    if *c + 4 > buf.len() { return Err(std::io::Error::other("EOF reading u32")); }
    let v = u32::from_le_bytes(buf[*c..*c + 4].try_into().unwrap());
    *c += 4;
    Ok(v)
}
#[inline]
fn read_f32(buf: &[u8], c: &mut usize) -> std::io::Result<f32> {
    if *c + 4 > buf.len() { return Err(std::io::Error::other("EOF reading f32")); }
    let v = f32::from_le_bytes(buf[*c..*c + 4].try_into().unwrap());
    *c += 4;
    Ok(v)
}
#[inline]
fn read_vec3(buf: &[u8], c: &mut usize) -> std::io::Result<Vec3> {
    let x = read_f32(buf, c)?;
    let y = read_f32(buf, c)?;
    let z = read_f32(buf, c)?;
    Ok(Vec3::new(x, y, z))
}
#[inline]
fn read_quat(buf: &[u8], c: &mut usize) -> std::io::Result<Quat> {
    let x = read_f32(buf, c)?;
    let y = read_f32(buf, c)?;
    let z = read_f32(buf, c)?;
    let w = read_f32(buf, c)?;
    Ok(Quat::from_xyzw(x, y, z, w))
}
/// Bounds-checked slice of `len` bytes; advances the cursor. Used for `tex_bytes`.
#[inline]
fn read_bytes_slice<'a>(buf: &'a [u8], c: &mut usize, len: usize) -> std::io::Result<&'a [u8]> {
    if *c + len > buf.len() { return Err(std::io::Error::other("EOF reading byte run")); }
    let s = &buf[*c..*c + len];
    *c += len;
    Ok(s)
}

/// Validate an untrusted element count BEFORE allocating. A file claiming `count`
/// elements each occupying at least `min_elem_bytes` is impossible if that
/// exceeds the bytes actually remaining — reject it as a clean `Err` instead of
/// attempting a multi-GB `Vec::with_capacity` that would abort the process.
/// (Copied from `colony_save_load.rs::checked_count`.)
fn checked_count(
    count:          u32,
    min_elem_bytes: usize,
    cursor:         usize,
    total:          usize,
    what:           &str,
) -> std::io::Result<usize> {
    let remaining    = total.saturating_sub(cursor);
    let max_possible = remaining / min_elem_bytes.max(1);
    if count as usize > max_possible {
        return Err(std::io::Error::other(format!(
            "{what}: implausible count {count} ({remaining} byte(s) left, \u{2264} {max_possible} possible) \
             \u{2014} file is corrupt or from an unsupported version"
        )));
    }
    Ok(count as usize)
}

/// Parse the 8-byte magic into a version number, accepting exactly the
/// `b"AEONSW0N"` family for `N` in `1..=MAX_WORLD_VERSION`. `None` for any foreign
/// / unsupported magic (the caller turns that into a clean `Err`).
fn parse_world_version(magic: &[u8]) -> Option<u8> {
    let bytes: &[u8; 8] = magic.try_into().ok()?;
    if &bytes[..5] != b"AEONS" { return None; }
    if bytes[5] != b'W' { return None; }
    // Two ASCII decimal digits in bytes 6..8 (e.g. "01" → 1).
    let mut version: u32 = 0;
    for &d in &bytes[6..8] {
        if !d.is_ascii_digit() { return None; }
        version = version * 10 + u32::from(d - b'0');
    }
    if (1..=u32::from(MAX_WORLD_VERSION)).contains(&version) {
        Some(version as u8)
    } else {
        None
    }
}


// ── Deserialise ──────────────────────────────────────────────────────────────────

fn read_world_bytes(bytes: &[u8]) -> std::io::Result<WorldData> {
    let total = bytes.len();

    if total < 8 {
        return Err(std::io::Error::other("world file too short — missing magic"));
    }
    if parse_world_version(&bytes[0..8]).is_none() {
        return Err(std::io::Error::other("world magic/version mismatch"));
    }
    let mut c = 8usize; // cursor positioned just past the 8-byte magic

    let map_x = read_f32(bytes, &mut c)?;
    let map_z = read_f32(bytes, &mut c)?;

    let tex_width  = read_u32(bytes, &mut c)?;
    let tex_height = read_u32(bytes, &mut c)?;
    let tex_format = read_u32(bytes, &mut c)?;
    let tex_len    = read_u32(bytes, &mut c)?;

    if tex_format != TEX_FORMAT_RGBA8_SRGB {
        return Err(std::io::Error::other(format!(
            "unsupported tex_format {tex_format} (only {TEX_FORMAT_RGBA8_SRGB} = Rgba8UnormSrgb)"
        )));
    }
    // Cross-check the declared length against the dimensions (u64 to avoid overflow).
    if tex_width as u64 * tex_height as u64 * 4 != tex_len as u64 {
        return Err(std::io::Error::other(
            "tex_len does not match tex_width*tex_height*4 — corrupt file",
        ));
    }
    let tex_len = checked_count(tex_len, 1, c, total, "tex_bytes")?;
    let tex_bytes = read_bytes_slice(bytes, &mut c, tex_len)?.to_vec();

    let mesh_count = read_u32(bytes, &mut c)?;
    // 44 = 3·4 translation + 4·4 rotation + 3·4 scale + 4 vert_count + 4 index_count
    // (the minimum bytes a mesh occupies even with zero verts/indices).
    let mesh_count = checked_count(mesh_count, 44, c, total, "mesh count")?;
    let mut meshes = Vec::with_capacity(mesh_count);

    for _ in 0..mesh_count {
        let translation = read_vec3(bytes, &mut c)?;
        let rotation    = read_quat(bytes, &mut c)?;
        let scale       = read_vec3(bytes, &mut c)?;
        let transform   = Transform { translation, rotation, scale };

        // 32 = pos 12 + normal 12 + uv 8 (the three consecutive runs per vertex).
        let vert_count = read_u32(bytes, &mut c)?;
        let vert_count = checked_count(vert_count, 32, c, total, "vertices")?;
        let mut positions = Vec::with_capacity(vert_count);
        let mut normals   = Vec::with_capacity(vert_count);
        let mut uv0       = Vec::with_capacity(vert_count);
        for _ in 0..vert_count {
            let x = read_f32(bytes, &mut c)?;
            let y = read_f32(bytes, &mut c)?;
            let z = read_f32(bytes, &mut c)?;
            positions.push([x, y, z]);
        }
        for _ in 0..vert_count {
            let x = read_f32(bytes, &mut c)?;
            let y = read_f32(bytes, &mut c)?;
            let z = read_f32(bytes, &mut c)?;
            normals.push([x, y, z]);
        }
        for _ in 0..vert_count {
            let u = read_f32(bytes, &mut c)?;
            let v = read_f32(bytes, &mut c)?;
            uv0.push([u, v]);
        }

        let index_count = read_u32(bytes, &mut c)?;
        let index_count = checked_count(index_count, 4, c, total, "indices")?;
        let mut indices = Vec::with_capacity(index_count);
        for _ in 0..index_count {
            indices.push(read_u32(bytes, &mut c)?);
        }
        // Indices are consumed three-at-a-time as triangles; a misaligned count
        // would silently drop the trailing partial triangle. Fail cleanly instead.
        if indices.len() % 3 != 0 {
            return Err(std::io::Error::other(format!(
                "index_count {} is not a multiple of 3 — corrupt file",
                indices.len()
            )));
        }
        // Every index must address a real vertex; reject out-of-range values with a
        // clean Err rather than uploading garbage indices to the GPU (out-of-bounds
        // vertex fetch). positions.len() == vert_count here.
        let n = positions.len() as u32;
        for &i in &indices {
            if i >= n {
                return Err(std::io::Error::other(format!(
                    "index {i} out of range (vert_count {n})"
                )));
            }
        }

        meshes.push(WorldMeshEntry { transform, positions, normals, uv0, indices });
    }

    Ok(WorldData {
        map_x,
        map_z,
        texture: PaintTextureData { width: tex_width, height: tex_height, bytes: tex_bytes },
        meshes,
    })
}


// ── Tests ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod world_format_tests {
    use super::*;

    fn sample() -> WorldData {
        // One submesh: 2 triangles, 4 verts, a non-identity transform.
        let entry = WorldMeshEntry {
            transform: Transform {
                translation: Vec3::new(1.5, -2.0, 3.25),
                rotation:    Quat::from_rotation_y(0.7),
                scale:       Vec3::new(2.0, 1.0, 0.5),
            },
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            normals: vec![
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            uv0: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
        };
        // 4×4 RGBA texture (tex_len = 64) with distinct bytes.
        let bytes: Vec<u8> = (0..64u32).map(|i| (i * 3 % 251) as u8).collect();
        WorldData {
            map_x: 512.0,
            map_z: 384.0,
            texture: PaintTextureData { width: 4, height: 4, bytes },
            meshes: vec![entry],
        }
    }

    #[test]
    fn round_trip() {
        let d = sample();
        let buf = write_world_bytes(&d);
        let r = read_world_bytes(&buf).expect("round-trip should decode");

        assert_eq!(r.map_x, d.map_x);
        assert_eq!(r.map_z, d.map_z);
        assert_eq!(r.texture.width, d.texture.width);
        assert_eq!(r.texture.height, d.texture.height);
        assert_eq!(r.texture.bytes, d.texture.bytes);
        assert_eq!(r.meshes.len(), d.meshes.len());

        let a = &r.meshes[0];
        let b = &d.meshes[0];
        // Geometry bit-exact.
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.normals, b.normals);
        assert_eq!(a.uv0, b.uv0);
        assert_eq!(a.indices, b.indices);
        // Transform within float epsilon.
        assert!((a.transform.translation - b.transform.translation).length() < 1e-6);
        assert!((a.transform.scale - b.transform.scale).length() < 1e-6);
        assert!(a.transform.rotation.angle_between(b.transform.rotation) < 1e-6);
    }

    #[test]
    fn bad_magic() {
        let mut buf = write_world_bytes(&sample());
        buf[5] = b'X'; // corrupt the 'W' marker
        assert!(read_world_bytes(&buf).is_err());
    }

    #[test]
    fn truncated() {
        let buf = write_world_bytes(&sample());
        // Half a buffer must error cleanly, never panic.
        assert!(read_world_bytes(&buf[..buf.len() / 2]).is_err());
    }

    #[test]
    fn implausible_count() {
        // Valid magic + map size + an empty texture, then a huge mesh_count with a
        // short buffer: the checked_count guard must reject it (no multi-GB alloc).
        let mut buf = Vec::new();
        buf.extend_from_slice(WORLD_MAGIC);
        buf.extend_from_slice(&0f32.to_le_bytes()); // map_x
        buf.extend_from_slice(&0f32.to_le_bytes()); // map_z
        buf.extend_from_slice(&0u32.to_le_bytes()); // tex_width
        buf.extend_from_slice(&0u32.to_le_bytes()); // tex_height
        buf.extend_from_slice(&TEX_FORMAT_RGBA8_SRGB.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // tex_len
        buf.extend_from_slice(&u32::MAX.to_le_bytes()); // mesh_count
        assert!(read_world_bytes(&buf).is_err());
    }

    #[test]
    fn bad_tex_format() {
        let mut buf = write_world_bytes(&sample());
        // tex_format sits at offset: 8 magic + 4 map_x + 4 map_z + 4 w + 4 h = 24.
        buf[24..28].copy_from_slice(&1u32.to_le_bytes());
        assert!(read_world_bytes(&buf).is_err());
    }

    #[test]
    fn bad_tex_len_mismatch() {
        let mut buf = write_world_bytes(&sample());
        // tex_len sits at offset 28 (after tex_format). Patch to a wrong value.
        buf[28..32].copy_from_slice(&999u32.to_le_bytes());
        assert!(read_world_bytes(&buf).is_err());
    }

    // index_count for the single sample mesh sits at:
    //   header 100 = magic 8 + map_x 4 + map_z 4 + tex_w 4 + tex_h 4
    //                + tex_format 4 + tex_len 4 + tex_bytes 64 + mesh_count 4
    //   + transform 40 (vec3 12 + quat 16 + vec3 12)
    //   + vert_count 4 + positions 48 (4×12) + normals 48 + uv0 32 (4×8)
    //   = 272.
    const SAMPLE_INDEX_COUNT_OFFSET: usize = 272;

    #[test]
    fn bad_index_count_not_multiple_of_3() {
        let mut buf = write_world_bytes(&sample());
        // 6 valid indices written; claim 7 (misaligned). The buffer is now short by
        // one index, so this could fail either on the multiple-of-3 check or the
        // truncated read — both are clean Err, which is all we require.
        buf[SAMPLE_INDEX_COUNT_OFFSET..SAMPLE_INDEX_COUNT_OFFSET + 4]
            .copy_from_slice(&7u32.to_le_bytes());
        assert!(read_world_bytes(&buf).is_err());
    }

    #[test]
    fn bad_index_out_of_range() {
        let mut d = sample();
        // 4 verts exist (0..=3); 9 is out of range.
        d.meshes[0].indices = vec![0, 1, 9, 0, 2, 3];
        let buf = write_world_bytes(&d);
        assert!(read_world_bytes(&buf).is_err());
    }
}
