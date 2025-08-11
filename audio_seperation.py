import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from sklearn.decomposition import FastICA
from itertools import permutations
from scipy.stats import pearsonr
import os
import random
import seaborn as sns
import warnings
import urllib.request
import zipfile
warnings.filterwarnings('ignore')

# Configuration for better plots
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

def get_output_directory():
    """Gets output directory from user at the beginning"""
    print("🔧 OUTPUT DIRECTORY CONFIGURATION")
    print("=" * 40)
    print("Where would you like to save the results?")
    print("1. Use default 'output' directory")
    print("2. Specify custom directory")
    
    while True:
        try:
            choice = int(input("Choose option (1-2): "))
            if choice == 1:
                output_dir = 'output'
                break
            elif choice == 2:
                output_dir = input("Enter directory path (without quotes): ").strip()
                # Remove quotes if user included them
                output_dir = output_dir.strip('"').strip("'")
                
                if not output_dir:
                    print("⚠️  Please enter a valid directory path")
                    continue
                
                # Convert forward slashes to backslashes on Windows if needed
                output_dir = os.path.normpath(output_dir)
                break
            else:
                print("⚠️  Please enter 1 or 2")
        except ValueError:
            print("⚠️  Please enter a valid integer")
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        abs_path = os.path.abspath(output_dir)
        print(f"✅ Output directory set to: {abs_path}")
        print(f"📁 All results will be saved here: {abs_path}\n")
        return output_dir
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        print("💡 Tip: Enter the path without quotes")
        print("📝 Example: C:\\Users\\username\\Documents\\my_folder")
        print("🔄 Using default 'output' directory instead")
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        abs_path = os.path.abspath(output_dir)
        print(f"✅ Fallback directory: {abs_path}\n")
        return output_dir


def download_fsdd():
    """Downloads and extracts the FSDD dataset"""
    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip"
    filename = "fsdd.zip"
    
    if not os.path.exists("free-spoken-digit-dataset-master"):
        print("Downloading FSDD dataset...")
        urllib.request.urlretrieve(url, filename)
        
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()
        
        os.remove(filename)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Dataset already available!")

class AudioProcessor:
    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path):
        """Loads an audio file"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def normalize_length(self, audios, target_length=None):
        """Normalizes the length of audio signals"""
        if target_length is None:
            target_length = min([len(audio) for audio in audios])
        
        normalized_audios = []
        for audio in audios:
            if len(audio) > target_length:
                # Truncate
                normalized_audios.append(audio[:target_length])
            else:
                # Pad with zeros
                padded = np.pad(audio, (0, target_length - len(audio)), 'constant')
                normalized_audios.append(padded)
        
        return normalized_audios
    
    def create_multi_source_mixture(self, signals, mixing_type='random'):
        """Creates a mixture of multiple signals"""
        n_sources = len(signals)
        
        # Normalize lengths
        normalized_signals = self.normalize_length(signals)
        signal_matrix = np.array(normalized_signals)
        
        # Generate mixing matrix based on type
        mixing_matrix = self.generate_mixing_matrix(n_sources, mixing_type)
        
        # Create mixtures
        mixed_signals = mixing_matrix @ signal_matrix
        
        return mixed_signals, normalized_signals, mixing_matrix
    
    def generate_mixing_matrix(self, n_sources, mixing_type='random'):
        """Generate different types of mixing matrices"""
        np.random.seed(42)
        
        if mixing_type == 'random':
            # Random mixing coefficients
            mixing_matrix = np.random.random((n_sources, n_sources))
            mixing_matrix = mixing_matrix / np.sum(mixing_matrix, axis=1, keepdims=True)
            
        elif mixing_type == 'balanced':
            # Equal contributions with small variations
            base_value = 1.0 / n_sources
            mixing_matrix = np.full((n_sources, n_sources), base_value)
            # Add small random variations
            variations = np.random.normal(0, 0.1, (n_sources, n_sources))
            mixing_matrix += variations
            mixing_matrix = np.abs(mixing_matrix)
            mixing_matrix = mixing_matrix / np.sum(mixing_matrix, axis=1, keepdims=True)
            
        elif mixing_type == 'weighted':
            # Decreasing contributions by distance
            mixing_matrix = np.zeros((n_sources, n_sources))
            for i in range(n_sources):
                for j in range(n_sources):
                    distance = abs(i - j) + 1
                    mixing_matrix[i, j] = 1.0 / distance
            # Normalize rows
            mixing_matrix = mixing_matrix / np.sum(mixing_matrix, axis=1, keepdims=True)
            
        elif mixing_type == 'complex':
            # Sinusoidal mixing pattern
            mixing_matrix = np.zeros((n_sources, n_sources))
            for i in range(n_sources):
                for j in range(n_sources):
                    angle = 2 * np.pi * i * j / n_sources
                    mixing_matrix[i, j] = abs(np.sin(angle) + 1)
            # Normalize rows
            mixing_matrix = mixing_matrix / np.sum(mixing_matrix, axis=1, keepdims=True)
            
        return mixing_matrix

class MultiSourceICA:
    def __init__(self, n_components=None, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_transform(self, mixed_signals):
        """Applies ICA to separate mixed signals"""
        try:
            # Initialize ICA
            ica = FastICA(
                n_components=self.n_components,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=42
            )
            
            # Apply ICA
            separated_signals = ica.fit_transform(mixed_signals.T).T
            
            return separated_signals
            
        except Exception as e:
            print(f"ICA separation failed: {e}")
            return None
    
    def find_best_permutation(self, original_signals, separated_signals):
        """Finds the best permutation to match separated signals with originals"""
        n_sources = len(original_signals)
        best_correlation = -1
        best_perm = None
        best_correlations = None
        
        for perm in permutations(range(n_sources)):
            total_correlation = 0
            correlations = []
            
            for i in range(n_sources):
                corr, _ = pearsonr(original_signals[i], separated_signals[perm[i]])
                correlations.append(abs(corr))
                total_correlation += abs(corr)
            
            if total_correlation > best_correlation:
                best_correlation = total_correlation
                best_perm = perm
                best_correlations = correlations
        
        return best_perm, best_correlations, best_correlation / n_sources
    
    def evaluate_multi_source_separation(self, original_signals, separated_signals):
        """Evaluates the quality of multi-source separation"""
        try:
            n_sources = len(original_signals)
            
            # Find the best permutation
            best_perm, correlations, mean_corr = self.find_best_permutation(
                original_signals, separated_signals
            )
            
            # Create correlation matrix
            correlation_matrix = np.zeros((n_sources, n_sources))
            for i in range(n_sources):
                for j in range(n_sources):
                    try:
                        corr, _ = pearsonr(original_signals[i], separated_signals[j])
                        correlation_matrix[i, j] = abs(corr)
                    except:
                        correlation_matrix[i, j] = 0
            
            return {
                'best_permutation': best_perm,
                'correlations': correlations,
                'mean_correlation': mean_corr,
                'min_correlation': min(correlations),
                'max_correlation': max(correlations),
                'correlation_matrix': correlation_matrix
            }
            
        except Exception as e:
            return {'error': str(e)}

def plot_multi_source_waveforms(signals, titles, save_path=None):
    """Plots multiple waveforms"""
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(15, 3*n_signals))
    
    if n_signals == 1:
        axes = [axes]
    
    for i, (signal, title) in enumerate(zip(signals, titles)):
        axes[i].plot(signal, linewidth=0.8)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Waveforms saved to: {save_path}")
    plt.show()

def plot_multi_source_spectrograms(signals, titles, save_path=None):
    """Plots spectrograms for multiple signals"""
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals//3 + (1 if n_signals%3 else 0), 3, 
                            figsize=(18, 4*(n_signals//3 + (1 if n_signals%3 else 0))))
    
    if n_signals <= 3:
        axes = axes.reshape(1, -1) if n_signals > 1 else [[axes]]
    
    for i, (signal, title) in enumerate(zip(signals, titles)):
        row = i // 3
        col = i % 3
        
        if n_signals == 1:
            ax = axes
        else:
            ax = axes[row, col] if n_signals > 3 else axes[col]
        
        D = librosa.stft(signal)
        magnitude = np.abs(D)
        db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        img = librosa.display.specshow(db, sr=8000, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Hide empty subplots
    for i in range(len(signals), (n_signals//3 + (1 if n_signals%3 else 0)) * 3):
        if n_signals > 3:
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectrograms saved to: {save_path}")
    plt.show()

def visualize_correlation_matrix(correlation_matrix, best_permutation, n_sources, save_path=None):
    """Visualizes the correlation matrix with best permutation highlighted"""
    plt.figure(figsize=(10, 8))
    
    # Create annotation matrix
    annot = np.round(correlation_matrix, 3)
    
    # Create mask for best permutation
    mask = np.ones_like(correlation_matrix, dtype=bool)
    for i, j in enumerate(best_permutation):
        mask[i, j] = False
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, annot=annot, cmap='RdYlGn', center=0.5,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                mask=None)
    
    # Highlight best permutation
    for i, j in enumerate(best_permutation):
        plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))
    
    plt.title(f'Correlation Matrix - {n_sources} Sources\n(Blue boxes show optimal matching)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Separated Signal Index')
    plt.ylabel('Original Signal Index')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to: {save_path}")
    plt.show()

def plot_mixing_matrix(mixing_matrix, save_path=None):
    """Visualizes the mixing matrix"""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(mixing_matrix, annot=True, cmap='viridis', 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Mixing Matrix Coefficients', fontsize=14, fontweight='bold')
    plt.xlabel('Source Signal Index')
    plt.ylabel('Mixed Signal Index')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mixing matrix saved to: {save_path}")
    plt.show()

def get_user_configuration():
    """Gets user configuration for the separation experiment"""
    print("📋 MULTI-SOURCE AUDIO SEPARATION CONFIGURATION")
    print("=" * 50)
    
    # Get number of sources
    while True:
        try:
            n_sources = int(input("How many audio sources do you want to separate? (2-10): "))
            if 2 <= n_sources <= 10:
                print(f"✅ Selected: {n_sources} sources\n")
                break
            else:
                print("⚠️  Please enter a number between 2 and 10")
        except ValueError:
            print("⚠️  Please enter a valid integer")
    
    # Get mixing type
    print("Available mixing types:")
    print("1. Random - Random mixing coefficients")
    print("2. Balanced - Equal contributions with variations")
    print("3. Weighted - Decreasing contributions by distance")
    print("4. Complex - Sinusoidal mixing pattern")
    
    mixing_types = ['random', 'balanced', 'weighted', 'complex']
    while True:
        try:
            mixing_choice = int(input("Choose mixing type (1-4): "))
            if 1 <= mixing_choice <= 4:
                mixing_type = mixing_types[mixing_choice - 1]
                print(f"✅ Selected: {mixing_type} mixing\n")
                break
            else:
                print("⚠️  Please enter a number between 1 and 4")
        except ValueError:
            print("⚠️  Please enter a valid integer")
    
    # Get file selection method
    print("File selection:")
    print("1. Random selection")
    print("2. Manual selection")
    
    while True:
        try:
            selection_choice = int(input("Choose selection method (1-2): "))
            if selection_choice in [1, 2]:
                random_selection = (selection_choice == 1)
                selection_method = "random" if random_selection else "manual"
                print(f"✅ Selected: {selection_method} selection\n")
                break
            else:
                print("⚠️  Please enter 1 or 2")
        except ValueError:
            print("⚠️  Please enter a valid integer")
    
    return {
        'n_sources': n_sources,
        'mixing_type': mixing_type,
        'random_selection': random_selection
    }

def run_separation_experiment(output_dir):
    """Runs the complete separation experiment"""
    
    # Get user configuration
    config = get_user_configuration()
    
    # Download dataset
    download_fsdd()
    
    # Initialize components
    processor = AudioProcessor(sample_rate=8000)
    separator = MultiSourceICA()
    
    # Get audio files
    dataset_path = "free-spoken-digit-dataset-master/recordings"
    all_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    
    n_sources = config['n_sources']
    mixing_type = config['mixing_type']
    
    if config['random_selection']:
        # Random selection
        selected_files = random.sample(all_files, n_sources)
        print("📁 Randomly selected files:")
        for i, file in enumerate(selected_files, 1):
            print(f"   {i}. {file}")
    else:
        # Manual selection
        print("📁 Available audio files (showing first 20):")
        for i, file in enumerate(all_files[:20], 1):
            print(f"   {i}. {file}")
        
        selected_files = []
        print(f"\nSelect {n_sources} files by entering their numbers:")
        for i in range(n_sources):
            while True:
                try:
                    choice = int(input(f"File {i+1}: ")) - 1
                    if 0 <= choice < len(all_files):
                        if all_files[choice] not in selected_files:
                            selected_files.append(all_files[choice])
                            print(f"   ✅ Added: {all_files[choice]}")
                            break
                        else:
                            print("   ⚠️  File already selected, choose another")
                    else:
                        print(f"   ⚠️  Please enter a number between 1 and {len(all_files)}")
                except ValueError:
                    print("   ⚠️  Please enter a valid integer")
    
    # Load audio signals
    print(f"\n🔄 Loading {n_sources} audio files...")
    signals = []
    file_info = []
    
    for file in selected_files:
        file_path = os.path.join(dataset_path, file)
        signal, sr = processor.load_audio(file_path)
        signals.append(signal)
        file_info.append(file)
        print(f"   ✅ Loaded: {file} (length: {len(signal)} samples)")
    
    # Create mixtures
    print(f"\n🔀 Creating {mixing_type} mixtures...")
    mixed_signals, original_signals, mixing_matrix = processor.create_multi_source_mixture(
        signals, mixing_type=mixing_type
    )
    
    # Apply ICA
    print("🔍 Applying ICA separation...")
    separator.n_components = n_sources
    separated_sources = separator.fit_transform(mixed_signals)
    
    if separated_sources is None:
        print("❌ Separation failed!")
        return None
    
    # Evaluate quality
    print("📊 Evaluating separation quality...")
    metrics = separator.evaluate_multi_source_separation(original_signals, separated_sources)
    
    if 'error' in metrics:
        print(f"❌ Evaluation failed: {metrics['error']}")
        return None
    
    # Display results
    print("\n🎯 SEPARATION RESULTS")
    print("=" * 30)
    print(f"Number of sources: {n_sources}")
    print(f"Mixing type: {mixing_type}")
    print(f"Mean correlation: {metrics['mean_correlation']:.4f}")
    print(f"Min correlation: {metrics['min_correlation']:.4f}")
    print(f"Max correlation: {metrics['max_correlation']:.4f}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    # Reorder separated signals according to best permutation
    reordered_separated = [separated_sources[metrics['best_permutation'][i]] for i in range(n_sources)]
    
    # Plot waveforms
    original_titles = [f"Original {i+1}: {info}" for i, info in enumerate(file_info)]
    mixed_titles = [f"Mixture {i+1}" for i in range(n_sources)]
    separated_titles = [f"Separated {i+1}" for i in range(n_sources)]
    
    plot_multi_source_waveforms(original_signals, original_titles, 
                                save_path=f'{output_dir}/original_waveforms_{n_sources}sources.png')
    plot_multi_source_waveforms(mixed_signals, mixed_titles,
                                save_path=f'{output_dir}/mixed_waveforms_{n_sources}sources.png')
    plot_multi_source_waveforms(reordered_separated, separated_titles,
                                save_path=f'{output_dir}/separated_waveforms_{n_sources}sources.png')
    
    # Plot spectrograms
    all_signals = list(original_signals) + list(mixed_signals) + reordered_separated
    all_titles = original_titles + mixed_titles + separated_titles
    plot_multi_source_spectrograms(all_signals, all_titles,
                                  save_path=f'{output_dir}/spectrograms_{n_sources}sources.png')
    
    # Plot correlation matrix and mixing matrix
    visualize_correlation_matrix(metrics['correlation_matrix'], metrics['best_permutation'], 
                                n_sources, save_path=f'{output_dir}/correlation_matrix_{n_sources}sources.png')
    plot_mixing_matrix(mixing_matrix, save_path=f'{output_dir}/mixing_matrix_{n_sources}sources.png')
    
    # Save audio files
    print("💾 Saving audio files...")
    for i, signal in enumerate(original_signals):
        sf.write(f'{output_dir}/original_{i+1}.wav', signal, 8000)
    
    for i, signal in enumerate(mixed_signals):
        sf.write(f'{output_dir}/mixed_{i+1}.wav', signal, 8000)
    
    for i, signal in enumerate(reordered_separated):
        sf.write(f'{output_dir}/separated_{i+1}.wav', signal, 8000)
    
    print("\n🎉 Experiment completed successfully!")
    print(f"📁 Check the '{output_dir}' directory for all generated files.")
    print(f"📍 Full path: {os.path.abspath(output_dir)}")
    
    return {
        'config': config,
        'metrics': metrics,
        'success': True
    }

if __name__ == "__main__":
    print("🎵 Multi-Source Audio Separation with ICA 🎵")
    print("=" * 50)
    print("Welcome to the interactive audio separation tool!")
    print("This tool will help you separate mixed audio sources using ICA.\n")
    
    # Get output directory FIRST
    output_dir = get_output_directory()
    
    # Run the experiment
    result = run_separation_experiment(output_dir)
    
    if result and result['success']:
        print("\n✅ SUCCESS! Your audio sources have been separated.")
        print(f"📁 All results saved in the '{output_dir}' directory")
        print(f"📍 Location: {os.path.abspath(output_dir)}")
        print("🎧 Listen to the separated audio files to hear the results!")
    else:
        print("\n❌ Experiment failed. Please try again with different settings.")
