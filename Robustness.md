1. Data Augmentation to Test Robustness
Goal
Demonstrate that your model’s predictions remain stable when the background noise is altered or when the signal is mildly transformed. If the model relies on invariant fault features, it should still classify correctly.

How to Implement
A. Add random underwater noise

Obtain external noise: Find datasets of underwater ambient sounds (e.g., from hydrophone repositories, ORCA, or marine biology recordings). Ensure they are different from your original recordings (different location, time, or conditions).

Mixing: For each test sample, mix in a random noise segment at various signal-to-noise ratios (SNRs). Start with a range (e.g., 0 dB to 20 dB) to simulate different levels of interference.

Measure: Compute the model’s accuracy (or precision per class) on these noisy versions. Plot accuracy vs. SNR. If accuracy stays high even at low SNRs, it’s a good sign.

B. Band-stop filtering

Identify background frequencies: Examine average spectrograms of your recordings to see where constant background noise dominates (e.g., low-frequency shipping hum, high-frequency hiss).

Apply notch filters: Remove those frequency bands (e.g., using a band-stop filter in Python’s scipy.signal). Be careful not to remove frequencies that are part of fault signatures.

Evaluate: Run the filtered test set through your model. If predictions remain largely unchanged, the model isn’t relying on those frequencies.

C. Time-stretch / pitch-shift

Use audio effects: Tools like librosa.effects.time_stretch or pitch_shift (keeping sampling rate constant). Apply mild stretching (e.g., ±5–10%) and pitch shifting (e.g., ±2 semitones) that preserve the essential temporal/spectral patterns of faults.

Caution: These transformations can alter fault signatures (e.g., a rhythmic pattern may change). If accuracy drops, it might indicate the model uses those temporal/spectral features, which is actually good—it shows sensitivity to fault characteristics. The key is whether the drop is graceful and whether the model still focuses on the right regions (which you can verify with Grad-CAM).

Measure: Track prediction consistency (e.g., the proportion of samples whose predicted class remains the same across transformations).

Presenting Results
Create a table or graph showing accuracy under each augmentation type and level.

Highlight that even with added noise or filtering, the model maintains high accuracy (if it does). If accuracy drops, discuss possible reasons (e.g., fault signatures are subtle and get masked).

Emphasize that the model’s invariance to irrelevant background changes is evidence of generalization.

1. Interpretability with Grad-CAM and Integrated Gradients
Goal
Visualize which parts of the input spectrogram the model uses for decision-making. If the model consistently highlights regions associated with propeller faults (e.g., harmonics, transients) and ignores background hum, you have strong qualitative evidence.

How to Implement
A. Grad-CAM for spectrogram-based CNNs

If your model uses a CNN (e.g., on mel-spectrograms), you can apply Grad-CAM.

Tools:

For PyTorch: pytorch-grad-cam library.

For TensorFlow/Keras: tf-keras-vis or custom implementation.

Procedure:

Select a trained model and a test sample.
Compute the Grad-CAM heatmap for the target class (the class the model predicted).
Overlay the heatmap on the spectrogram to see which time-frequency bins contributed most.
What to look for:

The heatmap should align with known fault signatures (e.g., periodic patterns, specific frequency bands).

Constant background regions (e.g., steady low-frequency noise) should have low activation.

For different fault classes, the heatmaps should highlight distinct patterns.

B. Integrated Gradients for any model

Integrated gradients attribute the prediction to input features. Works for any differentiable model.

Tools: captum for PyTorch, tf-explain or integrated-gradients for TensorFlow.

Procedure:

Compute attributions for each time-frequency bin of the spectrogram.
Visualize as a heatmap (similar to Grad-CAM).
Advantage: More theoretically grounded for feature attribution; can be used even if your model isn’t CNN-based (e.g., Transformer).

C. Combining with augmentation

Apply the same interpretability techniques to augmented versions of a sample (e.g., with added noise). If the heatmap still highlights the same fault regions despite background changes, it’s powerful evidence.

Presenting Results
Include side-by-side visualizations: original spectrogram, Grad-CAM heatmap overlay, and possibly the same for augmented samples.

Annotate the plots to point out where the model focuses (e.g., “The model consistently highlights the 200–400 Hz band where propeller cavitation occurs”).

Show examples for each fault class and for normal operation.

Discuss any cases where the model is misled by noise (if any) and what that implies.

Putting It Together for Your Defense
In your thesis or presentation, you can structure your verification as follows:

Motivation: Explain why it’s critical to ensure the model isn’t overfitting to background noise.

Methodology: Describe the two approaches—robustness testing via augmentation and interpretability via Grad-CAM/integrated gradients.

Results:

Show accuracy under various augmentations (tables/graphs).

Show heatmaps for representative samples, both clean and augmented.

Quantify the overlap between highlighted regions and known fault characteristics (if possible).

Discussion: Interpret the findings. For example, “The high accuracy under added noise and the consistent focus on fault-relevant frequencies indicate that our model has learned robust features.”

Limitations: Acknowledge that these tests are still based on the same original dataset; future work could involve cross-location validation.

By combining quantitative robustness checks with qualitative visual explanations, you’ll build a compelling case that your model generalizes beyond the specific background conditions of your recordings.

Good luck with your defense—you’ve got this!
