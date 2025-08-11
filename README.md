# Multi-Source-Audio-Separation-with-ICA
A comprehensive Python tool for separating mixed audio sources using Independent Component Analysis (ICA). This interactive application provides an intuitive interface to experiment with blind source separation techniques on audio data, with customizable output directories and comprehensive visualization.

## Contact & Support

GitHub Issues: Report bugs or request features

Email: bekim_hyseni@hotmail.fr

LinkedIn: https://ch.linkedin.com/in/bekim-hyseni

## Features
**Multi-source separation**: Support for 2-10 audio sources simultaneously\

**Multiple mixing strategies**: Random, balanced, weighted, and complex mixing patterns\

**Custom output directories**: Choose where to save your results with full path validation\

**Interactive configuration**: User-friendly prompts for all parameters\

**Comprehensive evaluation**: Correlation-based quality metrics with permutation optimization\

**Rich visualizations**: Waveforms, spectrograms, correlation matrices, and mixing matrices\

**Audio output**: Save original, mixed, and separated audio files in WAV format\

**FSDD dataset integration**: Automatic download of Free Spoken Digit Dataset\

**Robust error handling**: Graceful fallbacks and helpful error messages

## Requirements
### Dependencies
```bash
numpy
matplotlib
librosa
soundfile
scikit-learn
seaborn
pandas
scipy
```
### System requirements
Python 3.7+

Minimum 2GB RAM (4GB+ recommended for larger datasets)

Audio playback capability for result verification

## Quick Start
### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-source-audio-separation.git
cd multi-source-audio-separation
```
2. Install dependencies:
```bash
pip install numpy matplotlib librosa soundfile scikit-learn seaborn pandas scipy
```
### Basic Usage
1. Run the application:
```bash
python audio_separation.py
```
2. Follow the interactive prompts:
- Choose output directory (default or custom path)\
- Select number of sources (2-10)\
- Choose mixing type and file selection method\
- Let the algorithm work its magic!\

## Usage Example
### Example 1: Quick Start with Default Settings
```bash
🎵 Multi-Source Audio Separation with ICA 🎵
==================================================

🔧 OUTPUT DIRECTORY CONFIGURATION
========================================
Where would you like to save the results?
1. Use default 'output' directory
2. Specify custom directory
Choose option (1-2): 1
✅ Output directory set to: /your/path/output

How many audio sources do you want to separate? (2-10): 3
Select mixing type:
1. Random mixing
2. Balanced mixing
Choose option (1-2): 1
```
### Example 2: Custom Output Directory
```bash
🔧 OUTPUT DIRECTORY CONFIGURATION
========================================
Where would you like to save the results?
1. Use default 'output' directory
2. Specify custom directory
Choose option (1-2): 2

💡 Tips:
   - Don't use quotes around the path
   - Use forward slashes (/) or double backslashes (\\)
   - Example: C:/Users/username/Documents/my_results

Enter directory path: C:/Users/username/Documents/Audio_Results
✅ Output directory set to: C:\Users\username\Documents\Audio_Results
```
## Configuration Options
### Number of Sources
**Range**: 2-10 sources
**Recommendation**: Start with 2 or 3 sources for better separation quality
### Mixing type
1. **Random mixing**: General-purpose mixing with random coefficients
2. **Balanced mixing**: Equal contribution from all sources
3. **Weighted mixing**: User-defined source priorities
4. **Complex mixing**: Advanced mixing with phase relationships
### File selection
1. **Random selection**: Automatic random pickup from dataset
2. **Manual selection**: Choose specific audio files
## Technical Details
### Algorithm
**Method**: Independent Component Analysis (ICA) using FastICA
**Optimization**: Permutation-based correlation matching
**Preprocessing**: Audio normalization and length standardization
**Sample Rate**: 8kHz (optimized for speech signals)
### Quality Metrics 
**Correlation Analysis**: Pearson correlation between original and separated signals
**Signal-to-Noise Ratio**: Automatic SNR calculation
**Permutation Optimization**: Finds best source-separation matching

## Troubleshooting
### Common Issues
**❌ Directory Creation Error:**
```bash
❌ Error creating directory: [WinError 123] La syntaxe du nom de fichier...
```
**Solution**: Remove quotes from path and use forward slashes or double backslashes\
**❌ Audio File Not Found:**
```bash
❌ Error loading audio files
```
**Solution**: Ensure FSDD dataset is downloaded (automatic on first run)\
**❌ Memory Issues:**
```bash
MemoryError: Unable to allocate array
```
**Solution**: Reduce number of sources or use shorter audio files

## 🎯 Applications

🔬 **Audio Research**: Blind source separation experiments and algorithm comparison\
📚 **Educational**: Understanding ICA principles and signal processing\
🎧 **Audio Engineering**: Multi-channel audio analysis and processing\
🎙️ **Speech Processing**: Separating overlapping speech signals\
🎼 **Music Analysis**: Isolating instruments in mixed recordings

## Contributing
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

FSDD Dataset: Free Spoken Digit Dataset by Jakobovski\
LibROSA: Audio analysis library\
Scikit-learn: Machine learning algorithms including FastICA
