library(tidyverse)
library(tuneR)
library(seewave)
library(Rtsne)
library(fmsb)
library(corrplot)
library(scales)


audio_dir <- "ravdess/audio_speech_actors_01-24"

output_dir <- "ravdess_results_debug"
dir.create(output_dir, showWarnings = FALSE)
cat("Output directory:", output_dir, "\n")


parse_metadata <- function(filepath) {
  parts <- strsplit(basename(filepath), "-|\\.")[[1]]
  if (length(parts) < 7) {
    message("Skipping metadata parsing for: ", basename(filepath), " - Incorrect filename format.")
    return(NULL) 
  }
  emotion_labels <- c("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
  statement_labels <- c("kids", "dogs")
  intensity_labels <- c("normal", "strong")
  
  emotion_id <- suppressWarnings(as.integer(parts[3]))
  intensity_id <- suppressWarnings(as.integer(parts[4]))
  statement_id <- suppressWarnings(as.integer(parts[5]))
  repetition <- suppressWarnings(as.integer(parts[6]))
  actor <- suppressWarnings(as.integer(parts[7]))
  
  if (any(is.na(c(emotion_id, intensity_id, statement_id, repetition, actor))) ||
      emotion_id < 1 || emotion_id > length(emotion_labels) ||
      statement_id < 1 || statement_id > length(statement_labels) ||
      intensity_id < 1 || intensity_id > length(intensity_labels) ||
      actor < 1) {
    message("Skipping metadata parsing for: ", basename(filepath), " - Invalid component IDs or NA values.")
    return(NULL)
  }
  
  return(data.frame(
    file_path = filepath,
    emotion = emotion_labels[emotion_id],
    
    intensity = ifelse(emotion_id == 1, "normal", intensity_labels[intensity_id]),
    statement = statement_labels[statement_id],
    repetition = repetition,
    actor = actor,
    gender = ifelse(actor %% 2 == 0, "female", "male"),
    stringsAsFactors = FALSE
  ))
}

cat("Parsing metadata...\n")

wav_files <- list.files(path = audio_dir, pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)
if (length(wav_files) == 0) {
  stop("ERROR: No WAV files found in the specified directory: ", audio_dir)
}

metadata <- map_dfr(wav_files, parse_metadata)

if (nrow(metadata) == 0) {
  stop("ERROR: Metadata parsing failed for all files or no valid files found.")
}
cat("Parsed metadata for", nrow(metadata), "files.\n")


subset_df <- metadata

cat("Using", nrow(subset_df), "files for analysis (1 per emotion/gender combination)\n")
cat("   Files selected:\n")
print(subset_df %>% select(emotion, gender, file_path))



extract_features <- function(row) {
  wave <- readWave(row$file_path)
  
  if (wave@samp.rate < 8000) {
    message("WARNING: Original sample rate (", wave@samp.rate, " Hz) too low for ", basename(row$file_path), ". Skipping feature extraction for this file.")
    
    return(data.frame(duration=NA_real_, zcr=NA_real_, centroid=NA_real_, flatness=NA_real_, energy=NA_real_, pitch=NA_real_))
  }
  
  wave <- downsample(wave, samp.rate = 16000)
  audio <- mono(wave, "left") 
  duration <- length(audio@left) / audio@samp.rate
  fs <- audio@samp.rate 
  
  cat("--------------------------------------------------\n")
  cat("Processing:", basename(row$file_path), "| Duration:", round(duration, 2), "s | Sample Rate:", fs, "Hz\n")
  
  
  cat("  Calculating ZCR...\n")
  
  wl_zcr <- 1024
  zcr_val <- mean(zcr(audio, wl = wl_zcr, plot = FALSE))
  cat("    Raw ZCR mean:", zcr_val, "\n")
  
  
  cat("  Calculating Mean Spectrum (meanspec)...\n")
  
  wl_spec <- 1024
  mean_spec_data <- meanspec(audio, f = fs, wl = wl_spec, plot = FALSE)
  cat("    DEBUG: meanspec output structure:\n")
  
  cat("    DEBUG: First few rows of meanspec output (Frequency kHz, Amplitude):\n")
  print(head(mean_spec_data))
  
  cat("  Calculating Spectral Properties (specprop)...\n")
  
  spec_props <- specprop(mean_spec_data, f = fs)
  cat("    DEBUG: specprop output list:\n")
  print(spec_props) 
  
  centroid <- spec_props$cent
  
  flatness <- spec_props$sfm 
  
  if (is.null(flatness)) { 
    cat("    WARNING: specprop did not return 'sfm'. Setting flatness to NA.\n")
    flatness <- NA_real_
  }
  
  
  cat("  Calculating Energy (env)...\n")
  
  rms_env <- env(audio, f = fs, envt = "abs", plot = FALSE)
  rms <- mean(rms_env)
  cat("    Mean RMS Energy:", rms, "\n")
  
  
  cat("  Calculating Pitch (fund)...\n")
  
  wl_pitch <- 480
  
  
  pitch_track <- fund(audio, f = fs, ovlp = 50, threshold = 3,
                      fmax = 600, wl = wl_pitch, plot = FALSE)
  cat("    DEBUG: fund() output structure:\n")
  
  cat("    DEBUG: First few rows of fund() output (Time s, F0 kHz):\n")
  print(head(pitch_track))
  cat("    DEBUG: Last few rows of fund() output:\n")
  print(tail(pitch_track))
  
  
  mean_pitch_khz <- mean(pitch_track[, 2], na.rm = TRUE)
  cat("    Raw mean pitch (kHz, NA removed):", mean_pitch_khz, "\n")
  
  
  pitch_val <- if (is.nan(mean_pitch_khz)) {
    cat("    INFO: Mean pitch calculation resulted in NaN (likely no pitch detected). Setting pitch to NA.\n")
    NA_real_
  } else {
    mean_pitch_khz * 1000
  }
  
  
  cat("  >>> Extracted Features Summary:\n")
  cat(sprintf("      Duration: %.2f s\n", duration))
  cat(sprintf("      ZCR:      %.4f\n", zcr_val))
  cat(sprintf("      Centroid: %.2f Hz\n", centroid))
  
  cat(sprintf("      Flatness: %s\n", ifelse(is.na(flatness), "NA", sprintf("%.6f", flatness))))
  cat(sprintf("      Energy:   %.4f\n", rms))
  cat(sprintf("      Pitch:    %s Hz\n", ifelse(is.na(pitch_val), "NA", sprintf("%.2f", pitch_val))))
  cat("--------------------------------------------------\n")
  
  
  return(data.frame(duration, zcr = zcr_val, centroid, flatness, energy = rms, pitch = pitch_val))
}

cat("\nExtracting features (with detailed debug output)...\n")

features_list <- vector("list", nrow(subset_df))
for (i in 1:nrow(subset_df)) {
  cat("\nProcessing row", i, "of", nrow(subset_df), "- File:", basename(subset_df$file_path[i]), "\n")
  
  features_list[[i]] <- extract_features(subset_df[i,])
}
features <- bind_rows(features_list) 
df <- bind_cols(subset_df, features) 


cat("\n=== FEATURE DIAGNOSTICS ===\n")
features_to_check <- c("pitch", "zcr", "centroid", "flatness", "energy")

cat("Data Types:\n")
print(sapply(df[features_to_check], class))

cat("\nNA / NaN Counts per Feature:\n")
for (f in features_to_check) {
  
  n_na <- sum(is.na(df[[f]]) & !is.nan(df[[f]])) 
  n_nan <- sum(is.nan(df[[f]]))                 
  cat(sprintf(" - %-10s: %d NA, %d NaN\n", f, n_na, n_nan))
}

cat("\nFeature Summary (Raw Dataframe):\n")
print(summary(df[features_to_check]))


is_finite_matrix <- sapply(df[features_to_check], is.finite)

if (!is.matrix(is_finite_matrix)) {
  is_finite_matrix <- matrix(is_finite_matrix, nrow = nrow(df), dimnames = list(NULL, features_to_check))
}
df$valid_feature_count <- rowSums(is_finite_matrix, na.rm = TRUE) 

cat("\nValid (Finite) Feature Count per Row (out of", length(features_to_check), "):\n")
print(table(df$valid_feature_count)) 


feature_filename <- file.path(output_dir, "ravdess_subset_features_debug.csv")
write.csv(df, feature_filename, row.names = FALSE)
cat("Features saved to:", feature_filename, "\n")


summary_filename <- file.path(output_dir, "summary_debug.txt")
sink(summary_filename)
cat("RAVDESS Summary (Debug Run)\n=============================\n")
cat("Total files processed for feature extraction: ", nrow(df), "\n")
cat("Emotions included: ", paste(unique(df$emotion), collapse = ", "), "\n\n")
cat("Mean Features per Emotion (calculated only from rows with valid values for that feature):\n")


summary_stats <- df %>%
  group_by(emotion) %>%
  summarise(
    n_files = n(), 
    across(all_of(features_to_check), list(
      mean = ~mean(.x, na.rm = TRUE),
      n_valid = ~sum(!is.na(.x)) 
    )
    ),
    .groups = 'drop'
  ) %>%
  
  mutate(across(ends_with("_mean"), ~ round(.x, 2)))

print(summary_stats, width = 150) 
sink()
cat("Text summary saved to:", summary_filename, "\n")


min_valid_features <- 4 
cat("\nFiltering data: Keeping rows with at least", min_valid_features, "valid (finite) features.\n")
df_clean <- df %>%
  filter(valid_feature_count >= min_valid_features)

cat("Clean rows available for plotting:", nrow(df_clean), "\n")
if (nrow(df_clean) > 0) {
  cat("   Emotions remaining in clean data:", paste(sort(unique(df_clean$emotion)), collapse=", "), "\n")
}


if (nrow(df_clean) < 2) {
  cat("WARNING: Not enough clean data (", nrow(df_clean), " rows with >=", min_valid_features, " valid features) to generate plots. Stopping plotting section.\n")
} else {
  cat("\nProceeding with Plotting using", nrow(df_clean), "clean rows...\n")
  
  
  theme_clean <- theme_minimal(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = rel(1.2)),
          axis.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1, size=rel(0.9)), 
          legend.position = "bottom",
          plot.margin = margin(10, 10, 10, 10)) 
  
  
  unique_emotions_clean <- sort(unique(df_clean$emotion)) 
  palette <- scales::hue_pal()(length(unique_emotions_clean))
  
  emotion_colors <- setNames(palette, unique_emotions_clean)
  
  
  cat("Generating Pitch Plot...\n")
  
  pitch_summary <- df_clean %>%
    filter(!is.na(pitch)) %>% 
    group_by(emotion) %>%
    summarise(mean_pitch = mean(pitch, na.rm = TRUE), .groups = 'drop')
  
  if(nrow(pitch_summary) > 0) {
    p1 <- ggplot(pitch_summary, aes(x = reorder(emotion, -mean_pitch), y = mean_pitch, fill = emotion)) +
      geom_col(show.legend = FALSE) + 
      scale_fill_manual(values = emotion_colors) +
      labs(title = "Mean Pitch by Emotion (Clean Data)", y = "Mean Pitch (Hz)", x = "Emotion") +
      theme_clean
    ggsave(file.path(output_dir, "pitch_by_emotion.png"), plot = p1, width = 8, height = 6, dpi = 150)
    cat("Saved pitch_by_emotion.png\n")
  } else {
    cat("No valid pitch data found in the 'df_clean' subset to plot.\n")
  }
  
  
  cat("Generating Energy Plot...\n")
  
  energy_summary <- df_clean %>%
    filter(!is.na(energy)) %>% 
    group_by(emotion) %>%
    summarise(mean_energy = mean(energy, na.rm = TRUE), .groups = 'drop')
  
  if(nrow(energy_summary) > 0) {
    p2 <- ggplot(energy_summary, aes(x = reorder(emotion, -mean_energy), y = mean_energy, fill = emotion)) +
      geom_col(show.legend = FALSE) +
      scale_fill_manual(values = emotion_colors) +
      labs(title = "Mean Energy by Emotion (Clean Data)", y = "Mean Energy (RMS)", x = "Emotion") +
      theme_clean
    ggsave(file.path(output_dir, "energy_by_emotion.png"), plot = p2, width = 8, height = 6, dpi = 150)
    cat(" Saved energy_by_emotion.png\n")
  } else {
    cat("No valid energy data found in the 'df_clean' subset to plot.\n")
  }
  
  
  cat("Generating Radar Chart...\n")
  
  radar_data_raw <- df_clean %>%
    group_by(emotion) %>%
    
    summarise(across(all_of(features_to_check), ~ mean(.x, na.rm = TRUE)), .groups = 'drop')
  
  
  all_na_cols <- names(which(sapply(radar_data_raw, function(col) all(is.na(col)))))
  if(length(all_na_cols) > 0) {
    cat(" Columns with all NA values after averaging:", paste(all_na_cols, collapse=", "), "- removing them for radar chart.\n")
    radar_data_raw <- radar_data_raw %>% select(-all_of(all_na_cols))
  }
  
  radar_data <- radar_data_raw %>%
    select(where(~!all(is.na(.)))) %>% 
    drop_na() 
    
  radar_features <- setdiff(names(radar_data), "emotion")
  
  if (nrow(radar_data) >= 2 && length(radar_features) >= 3) {
    radar_data_formated <- radar_data %>% column_to_rownames("emotion")
      
    scaled_radar_values <- scale(radar_data_formated)
    
    scaled_radar_values[is.nan(scaled_radar_values)] <- 0
    
    max_val <- max(scaled_radar_values, na.rm = TRUE) * 1.1
    min_val <- min(scaled_radar_values, na.rm = TRUE) * 1.1
    
    min_val <- min(min_val, -0.1)
    max_val <- max(max_val, 0.1)
        
    radar_plot_data <- rbind(rep(max_val, length(radar_features)),
                             rep(min_val, length(radar_features)),
                             scaled_radar_values)
    radar_plot_data <- as.data.frame(radar_plot_data) 
    
    radar_emotions <- rownames(radar_data_formated)
    radar_colors <- emotion_colors[radar_emotions] 
    
    png(file.path(output_dir, "radar_features.png"), width = 800, height = 700, res = 100)
    radarchart(radar_plot_data,
               axistype = 1,            
               pcol = radar_colors, pfcol = scales::alpha(radar_colors, 0.3), plwd = 2, plty = 1,
               cglcol = "grey", cglty = 1, axislabcol = "grey", caxislabels = round(seq(min_val, max_val, length.out = 5),1), cglwd = 0.8,
               vlcex = 0.9, 
               title = paste("Radar Chart: Scaled Audio Features (Clean Data -", nrow(radar_data_formated), "Emotions)")
    )
    legend("topright",
           legend = radar_emotions,
           fill = scales::alpha(radar_colors, 0.3),
           border = radar_colors,
           bty = "n", pch = 20, pt.cex = 2, cex = 0.9, 
           title = "Emotion")
    dev.off()
    cat("Saved radar_features.png\n")
  } else {
    cat(" Not enough valid data rows (need >= 2) or features (need >= 3) after cleaning and NA removal for radar chart.\n")
    cat("       Rows available:", nrow(radar_data), "Features available:", length(radar_features), "\n")
  }
  
  
  cat("Generating Correlation Matrix...\n")
  
  cor_data <- df_clean %>%
    select(all_of(features_to_check)) %>%
    filter(if_all(everything(), is.finite)) 
  
  if (nrow(cor_data) >= 2 && ncol(cor_data) >= 2) {
    cor_matrix <- cor(cor_data)
    png(file.path(output_dir, "feature_correlation.png"), width = 700, height = 600, res = 100)
    corrplot(cor_matrix,
             method = "color", type = "upper", order = "hclust", 
             addCoef.col = "black", 
             tl.col = "black", tl.srt = 45, 
             diag = FALSE, 
             cl.cex = 0.8, number.cex = 0.8, tl.cex = 0.9, 
             title = "Feature Correlation Matrix (Clean Data)", mar=c(0,0,1,0)) 
    dev.off()
    cat(" Saved feature_correlation.png\n")
  } else {
    cat("Not enough finite data (need >= 2 rows and >= 2 columns) in 'df_clean' for correlation matrix.\n")
    cat("       Finite rows:", nrow(cor_data), "Finite columns:", ncol(cor_data), "\n")
  }
  
  
  cat("Running t-SNE...\n")
  
  tsne_ready_data <- df_clean %>%
    select(all_of(features_to_check)) %>%
    filter(if_all(everything(), is.finite))
  
  
  is_finite_matrix_clean <- sapply(df_clean[features_to_check], is.finite)
  
  if (!is.matrix(is_finite_matrix_clean)) {
    is_finite_matrix_clean <- matrix(is_finite_matrix_clean, nrow = nrow(df_clean), dimnames = list(NULL, features_to_check))
  }
  
  finite_rows_indices <- which(rowSums(is_finite_matrix_clean) == length(features_to_check))
  
  
  
  if(length(finite_rows_indices) != nrow(tsne_ready_data)){
    warning("Mismatch between identified finite rows and rows in tsne_ready_data. Check filtering logic.")
    
    if(nrow(df_clean) == nrow(tsne_ready_data)) {
      tsne_labels <- df_clean$emotion
      warning("Assuming label order is preserved as row counts match.")
    } else {
      tsne_labels <- rep("Unknown", nrow(tsne_ready_data)) 
      warning("Cannot reliably determine labels.")
    }
  } else {
    tsne_labels <- df_clean$emotion[finite_rows_indices]
  }
  
  
  if (nrow(tsne_ready_data) >= 3 && ncol(tsne_ready_data) >= 1) { 
    
    tsne_perplexity <- min(5, floor((nrow(tsne_ready_data) - 1) / 3))
    
    if (tsne_perplexity < 1) {
      cat("Not enough data points (", nrow(tsne_ready_data), ") for the minimum perplexity (1). Skipping t-SNE.\n")
    } else {
      cat("Running Rtsne with perplexity =", tsne_perplexity, "on", nrow(tsne_ready_data), "data points...\n")
      set.seed(42) 
      tsne_result <- Rtsne(tsne_ready_data,
                           dims = 2,
                           perplexity = tsne_perplexity,
                           verbose = FALSE,
                           check_duplicates = FALSE,
                           pca = (ncol(tsne_ready_data) > 50))
      
      tsne_plot_df <- as.data.frame(tsne_result$Y)
      colnames(tsne_plot_df) <- c("TSNE1", "TSNE2")
      
      if(length(tsne_labels) == nrow(tsne_plot_df)) {
        tsne_plot_df$emotion <- tsne_labels 
      } else {
        warning("Label vector length mismatch for t-SNE plot. Using default labels.")
        tsne_plot_df$emotion <- "Label Error"
      }
      
      p_tsne <- ggplot(tsne_plot_df, aes(x = TSNE1, y = TSNE2, color = emotion)) +
        geom_point(size = 3, alpha = 0.8) +
        scale_color_manual(values = emotion_colors) + 
        labs(title = "t-SNE Visualization of Audio Features (Clean Data)",
             x = "t-SNE Dimension 1", y = "t-SNE Dimension 2",
             color = "Emotion") +
        theme_clean +
        theme(legend.position = "right")
      
      ggsave(file.path(output_dir, "tsne_audio_features.png"), plot = p_tsne, width = 8, height = 6, dpi = 150)
      cat("Saved tsne_audio_features.png\n")
    }
  } else {
    cat("Not enough finite data rows (need >= 3) or columns (need >= 1) in 'df_clean' for t-SNE.\n")
    cat("       Available rows:", nrow(tsne_ready_data), "Available columns:", ncol(tsne_ready_data), "\n")
  }
} 

message("\nDONE! Debug run complete. All results saved in '", output_dir, "'")
message(" Check the console output and 'summary_debug.txt' for details on feature values and potential warnings.")