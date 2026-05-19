setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
simulation_df <- read.csv("simulation_dataset_19-05-2026-01-55-10.csv", sep = ";")

View(my_data)

herbivores <- simulation_df[simulation_df$trophic_class == "Herbivore", ]

plot(dopamine ~ entity_id, data = simulation_df,
     main = "My Scatterplot Title",
     xlab = "Entity ID",
     ylab = "Dopamine",
     col = "blue", 
     pch = 19) # 'pch = 19' makes the points solid circles instead of open circles
