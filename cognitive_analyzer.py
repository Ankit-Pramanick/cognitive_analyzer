# MemoTag Speech Intelligence Module - Cognitive Impairment Detection
# POC Implementation

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import whisper
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.whisper_model = whisper.load_model("base")
        
    def load_audio(self, file_path):
        """Load and normalize audio file"""
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio
    
    def denoise(self, audio):
        """Basic noise reduction"""
        # Simple noise reduction using spectral gating
        # In a production system, we'd use a more sophisticated method
        return audio  # Placeholder for actual denoising
    
    def transcribe(self, audio):
        """Convert speech to text using Whisper"""
        result = self.whisper_model.transcribe(audio)
        return result["text"]
    
    def extract_audio_features(self, audio):
        """Extract acoustic features from audio"""
        # Duration
        duration = librosa.get_duration(y=audio, sr=self.sample_rate)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Spectral features (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Zero crossing rate (related to voice quality)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Detect pauses
        # Simple pause detection using energy threshold
        pause_threshold = 0.01
        is_pause = rms < pause_threshold
        pauses = np.sum(is_pause)
        pause_ratio = np.sum(is_pause) / len(rms)
        
        features = {
            'duration': duration,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'pause_count': pauses,
            'pause_ratio': pause_ratio,
        }
        
        # Add MFCCs
        for i, (mean, std) in enumerate(zip(mfcc_means, mfcc_stds)):
            features[f'mfcc_{i+1}_mean'] = mean
            features[f'mfcc_{i+1}_std'] = std
            
        return features


class TextAnalyzer:
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.hesitation_markers = {'uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well'}
        
    def analyze_text(self, text):
        """Extract linguistic features from transcribed text"""
        # Clean text
        text = text.lower()
        
        # Tokenize
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        # Lexical features
        word_count = len(words)
        unique_words = len(set(words))
        
        # Type-token ratio (lexical diversity)
        ttr = unique_words / word_count if word_count > 0 else 0
        
        # Sentence structure
        sentence_count = len(sentences)
        words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Hesitation markers count
        hesitation_count = 0
        for word in words:
            if word in self.hesitation_markers:
                hesitation_count += 1
                
        # Check for incomplete sentences using regex
        incomplete_pattern = r'([^.!?]*[a-zA-Z][^.!?]*)?[.!?]'
        incomplete_sentences = len(re.findall(incomplete_pattern, text))
        
        # Word frequency analysis (could help detect repetitions)
        word_freq = Counter(words)
        most_common_word, most_common_count = word_freq.most_common(1)[0] if word_freq else ('', 0)
        repetition_ratio = most_common_count / word_count if word_count > 0 else 0
        
        features = {
            'word_count': word_count,
            'unique_words': unique_words,
            'type_token_ratio': ttr,
            'sentence_count': sentence_count,
            'words_per_sentence': words_per_sentence,
            'hesitation_count': hesitation_count,
            'hesitation_ratio': hesitation_count / word_count if word_count > 0 else 0,
            'incomplete_sentences': incomplete_sentences,
            'repetition_ratio': repetition_ratio
        }
        
        return features


class CognitiveAnalyzer:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.text_analyzer = TextAnalyzer()
        self.feature_data = []
        self.patient_ids = []
        
    def process_sample(self, file_path, patient_id):
        """Process a single audio sample and extract all features"""
        # Load and preprocess audio
        audio = self.audio_processor.load_audio(file_path)
        audio = self.audio_processor.denoise(audio)
        
        # Extract audio features
        audio_features = self.audio_processor.extract_audio_features(audio)
        
        # Transcribe audio to text
        transcription = self.audio_processor.transcribe(audio)
        
        # Extract text features
        text_features = self.text_analyzer.analyze_text(transcription)
        
        # Combine all features
        all_features = {**audio_features, **text_features}
        
        # Store results
        self.feature_data.append(all_features)
        self.patient_ids.append(patient_id)
        
        return transcription, all_features
    
    def analyze_all(self):
        """Analyze all processed samples using unsupervised methods"""
        if not self.feature_data:
            return "No data to analyze"
        
        # Convert to DataFrame
        df = pd.DataFrame(self.feature_data)
        df['patient_id'] = self.patient_ids
        
        # Standardize features
        feature_cols = [col for col in df.columns if col != 'patient_id']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        
        # 1. PCA for dimensionality reduction and feature importance
        pca = PCA(n_components=min(5, len(feature_cols)))
        pca_result = pca.fit_transform(scaled_features)
        
        # Get feature importance
        feature_importance = pd.DataFrame(
            data=pca.components_,
            columns=feature_cols
        )
        
        # 2. K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        df['cluster'] = clusters
        
        # 3. Anomaly detection
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = isolation_forest.fit_predict(scaled_features)
        df['anomaly'] = ['Anomaly' if a == -1 else 'Normal' for a in anomalies]
        
        # Combine results
        results = {
            'data': df,
            'pca_result': pca_result,
            'feature_importance': feature_importance,
            'anomaly_count': sum(1 for a in anomalies if a == -1)
        }
        
        return results
    
    def visualize_results(self, results):
        """Visualize the analysis results"""
        df = results['data']
        
        # 1. Plot PCA results with cluster coloring
        plt.figure(figsize=(10, 6))
        pca_df = pd.DataFrame(results['pca_result'][:, :2], columns=['PC1', 'PC2'])
        pca_df['cluster'] = df['cluster']
        pca_df['anomaly'] = df['anomaly']
        pca_df['patient_id'] = df['patient_id']
        
        plt.scatter(
            pca_df['PC1'], 
            pca_df['PC2'], 
            c=pca_df['cluster'],
            cmap='viridis', 
            alpha=0.8,
            s=100
        )
        
        # Mark anomalies with a red circle
        anomalies = pca_df[pca_df['anomaly'] == 'Anomaly']
        if not anomalies.empty:
            plt.scatter(
                anomalies['PC1'],
                anomalies['PC2'],
                s=200,
                linewidth=2,
                facecolors='none',
                edgecolors='red',
                label='Potential Cognitive Decline'
            )
        
        # Add patient IDs as labels
        for i, row in pca_df.iterrows():
            plt.annotate(row['patient_id'], 
                         (row['PC1'], row['PC2']),
                         xytext=(5, 5),
                         textcoords='offset points')
        
        plt.title('Patient Clustering by Speech Patterns')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Feature importance plot
        importance = results['feature_importance'].iloc[0].abs().sort_values(ascending=False)
        top_features = importance.head(10)
        
        plt.figure(figsize=(12, 6))
        top_features.plot(kind='bar')
        plt.title('Top 10 Important Features for Cognitive Assessment')
        plt.ylabel('Absolute Coefficient (PCA Component 1)')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return "Visualizations created successfully"


# Example usage
def run_demo():
    """Run a demonstration of the pipeline with example data"""
    analyzer = CognitiveAnalyzer()
    
    # In a real implementation, you would load actual audio files
    # For this POC, we'll simulate the feature extraction process
    
    # Simulated data for demonstration purposes
    simulated_patients = [
        {'id': 'patient_001', 'features': {
            'duration': 120, 'pitch_mean': 150, 'pitch_std': 30, 'energy_mean': 0.8, 
            'energy_std': 0.2, 'zcr_mean': 0.05, 'zcr_std': 0.01, 'pause_count': 15, 
            'pause_ratio': 0.1, 'mfcc_1_mean': 10, 'mfcc_1_std': 2, 'word_count': 200, 
            'unique_words': 120, 'type_token_ratio': 0.6, 'sentence_count': 18, 
            'words_per_sentence': 11.1, 'hesitation_count': 5, 'hesitation_ratio': 0.025,
            'incomplete_sentences': 2, 'repetition_ratio': 0.05
        }},
        {'id': 'patient_002', 'features': {
            'duration': 115, 'pitch_mean': 145, 'pitch_std': 28, 'energy_mean': 0.75, 
            'energy_std': 0.18, 'zcr_mean': 0.06, 'zcr_std': 0.012, 'pause_count': 12, 
            'pause_ratio': 0.08, 'mfcc_1_mean': 11, 'mfcc_1_std': 2.2, 'word_count': 210, 
            'unique_words': 130, 'type_token_ratio': 0.62, 'sentence_count': 20, 
            'words_per_sentence': 10.5, 'hesitation_count': 4, 'hesitation_ratio': 0.019,
            'incomplete_sentences': 1, 'repetition_ratio': 0.04
        }},
        {'id': 'patient_003', 'features': {
            'duration': 130, 'pitch_mean': 160, 'pitch_std': 35, 'energy_mean': 0.85, 
            'energy_std': 0.22, 'zcr_mean': 0.055, 'zcr_std': 0.011, 'pause_count': 10, 
            'pause_ratio': 0.07, 'mfcc_1_mean': 12, 'mfcc_1_std': 2.5, 'word_count': 220, 
            'unique_words': 140, 'type_token_ratio': 0.64, 'sentence_count': 21, 
            'words_per_sentence': 10.5, 'hesitation_count': 3, 'hesitation_ratio': 0.014,
            'incomplete_sentences': 0, 'repetition_ratio': 0.03
        }},
        {'id': 'patient_004', 'features': {
            'duration': 90, 'pitch_mean': 155, 'pitch_std': 40, 'energy_mean': 0.7, 
            'energy_std': 0.3, 'zcr_mean': 0.07, 'zcr_std': 0.02, 'pause_count': 25, 
            'pause_ratio': 0.25, 'mfcc_1_mean': 9, 'mfcc_1_std': 3, 'word_count': 150, 
            'unique_words': 70, 'type_token_ratio': 0.47, 'sentence_count': 12, 
            'words_per_sentence': 8.5, 'hesitation_count': 18, 'hesitation_ratio': 0.12,
            'incomplete_sentences': 5, 'repetition_ratio': 0.12
        }},
        {'id': 'patient_005', 'features': {
            'duration': 100, 'pitch_mean': 140, 'pitch_std': 20, 'energy_mean': 0.65, 
            'energy_std': 0.25, 'zcr_mean': 0.065, 'zcr_std': 0.018, 'pause_count': 30, 
            'pause_ratio': 0.28, 'mfcc_1_mean': 8, 'mfcc_1_std': 2.8, 'word_count': 140, 
            'unique_words': 65, 'type_token_ratio': 0.46, 'sentence_count': 10, 
            'words_per_sentence': 7.0, 'hesitation_count': 20, 'hesitation_ratio': 0.14,
            'incomplete_sentences': 6, 'repetition_ratio': 0.15
        }}
    ]
    
    # Add simulated data to the analyzer
    for patient in simulated_patients:
        analyzer.feature_data.append(patient['features'])
        analyzer.patient_ids.append(patient['id'])
    
    # Perform analysis
    results = analyzer.analyze_all()
    
    # Generate visualizations
    analyzer.visualize_results(results)
    
    # Print summary report
    print("\n--- Cognitive Assessment Summary ---")
    print(f"Total patients analyzed: {len(analyzer.patient_ids)}")
    print(f"Potential cognitive impairment cases detected: {results['anomaly_count']}")
    print("\nPatients flagged for further evaluation:")
    
    anomaly_patients = results['data'][results['data']['anomaly'] == 'Anomaly']
    for _, row in anomaly_patients.iterrows():
        print(f"- Patient ID: {row['patient_id']}")
    
    print("\nMost important indicators (based on PCA):")
    importance = results['feature_importance'].iloc[0].abs().sort_values(ascending=False)
    for feature, value in importance.head(5).items():
        print(f"- {feature}: {value:.3f}")
    
    return results


if __name__ == "__main__":
    run_demo()