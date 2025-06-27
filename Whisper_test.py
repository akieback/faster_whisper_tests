# Whisper Model Performance Comparison Script
# This script tests various Whisper model sizes on a sample audio file,
# measures their performance in terms of processing time, Word Error Rate (WER),
# and Character Error Rate (CER), and visualizes the results with charts.
# Optimized for Jetson Orin unified memory architecture.

# Required Libraries
import time
import matplotlib.pyplot as plt
import pandas as pd
from faster_whisper import WhisperModel, BatchedInferencePipeline
from jiwer import wer, cer
import numpy as np
import gc
import torch
import psutil
import os
from datetime import datetime

# Create timestamped log file for correlation with memory monitor
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timing_log_file = f"whisper_timing_log_{timestamp}.log"

def log_with_timestamp(message, log_file=timing_log_file):
    """Log message with precise timestamp for correlation with memory monitoring"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
    log_entry = f"[{current_time}] {message}"
    print(log_entry)
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')

# Initialize timing log
with open(timing_log_file, 'w') as f:
    f.write(f"# Whisper Model Testing Timeline - Jetson Orin\n")
    f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# Correlate with memory_usage logs for complete analysis\n\n")

# Reference transcript (ground truth)
reference_transcript = ""

# Different model sizes to test
model_sizes = [
    "tiny",
    "base", 
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2", 
    "large-v3",
    "distil-large-v3",
    "turbo",
    "large-v3-turbo"
]

# Audio file to transcribe
audio_file = ""

# Results storage with timing info
results = []

log_with_timestamp("=== WHISPER MODEL TESTING STARTED ===")
log_with_timestamp("Testing different Whisper model sizes on Jetson Orin (Unified Memory)")
log_with_timestamp(f"Audio file: {audio_file}")
log_with_timestamp(f"Models to test: {', '.join(model_sizes)}")
log_with_timestamp("="*70)

for i, model_size in enumerate(model_sizes):
    log_with_timestamp(f"\n=== TESTING MODEL {i+1}/{len(model_sizes)}: {model_size} ===")
    
    try:
        # Log cleanup phase
        log_with_timestamp(f"[{model_size}] Starting memory cleanup and preparation...")
        
        # Force garbage collection and clear cache before measuring baseline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Wait a moment for system to stabilize
        time.sleep(1)
        log_with_timestamp(f"[{model_size}] System stabilized, measuring baseline memory...")
        
        # Get initial unified memory usage (Jetson Orin uses shared RAM for CPU+GPU)
        initial_unified_memory = psutil.virtual_memory().used / 1024**3
        log_with_timestamp(f"[{model_size}] Baseline unified memory: {initial_unified_memory:.2f} GB")
        
        # Log model loading start
        model_load_start = time.time()
        log_with_timestamp(f"[{model_size}] ⏳ LOADING MODEL - START")
        
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        batched_model = BatchedInferencePipeline(model=model)
        
        # Wait for model to fully load and stabilize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(0.5)
        
        model_load_end = time.time()
        model_load_time = model_load_end - model_load_start
        log_with_timestamp(f"[{model_size}] ✅ MODEL LOADED - END (Load time: {model_load_time:.2f}s)")
        
        # Get unified memory usage after loading (on Jetson Orin this includes GPU allocation)
        post_load_unified_memory = psutil.virtual_memory().used / 1024**3
        log_with_timestamp(f"[{model_size}] Post-load unified memory: {post_load_unified_memory:.2f} GB")
        
        # Calculate memory used by model (unified memory for both CPU and GPU)
        unified_memory_used = max(0, post_load_unified_memory - initial_unified_memory)
        
        # If memory shows negative, it's likely due to OS fluctuations
        if post_load_unified_memory - initial_unified_memory < 0:
            log_with_timestamp(f"[{model_size}] ⚠️ Memory fluctuation detected (OS background processes)")
            unified_memory_used = 0  # Set to 0 as we can't accurately measure
        
        log_with_timestamp(f"[{model_size}] 📊 Unified Memory used: {unified_memory_used:.2f} GB (CPU+GPU shared)")
        
        # Log transcription start
        log_with_timestamp(f"[{model_size}] 🎯 TRANSCRIPTION - START")
        start_time = time.time()
        
        # Transcribe audio
        segments, info = batched_model.transcribe(audio_file, batch_size=16, beam_size=1)
        
        # Combine all segments into one transcript
        transcript = " ".join([segment.text.strip() for segment in segments])
        
        # End timing
        end_time = time.time()
        transcription_time = end_time - start_time
        log_with_timestamp(f"[{model_size}] ✅ TRANSCRIPTION - END (Processing time: {transcription_time:.2f}s)")
        
        # Calculate WER (Word Error Rate)
        word_error_rate = wer(reference_transcript, transcript)
        
        # Calculate CER (Character Error Rate) as additional metric
        char_error_rate = cer(reference_transcript, transcript)
        
        # Log results
        log_with_timestamp(f"[{model_size}] 📈 RESULTS:")
        log_with_timestamp(f"[{model_size}]   WER: {word_error_rate:.4f} ({word_error_rate*100:.2f}%)")
        log_with_timestamp(f"[{model_size}]   CER: {char_error_rate:.4f} ({char_error_rate*100:.2f}%)")
        log_with_timestamp(f"[{model_size}]   Transcript: {transcript}")
        
        # Store results with unified memory and timing info
        results.append({
            'model_size': model_size,
            'processing_time': transcription_time,
            'model_load_time': model_load_time,
            'wer': word_error_rate,
            'cer': char_error_rate,
            'unified_memory_gb': unified_memory_used,
            'transcript': transcript,
            'load_start_time': datetime.fromtimestamp(model_load_start).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'load_end_time': datetime.fromtimestamp(model_load_end).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'transcription_start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'transcription_end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        })
        
        # Log cleanup start
        log_with_timestamp(f"[{model_size}] 🧹 MEMORY CLEANUP - START")
        cleanup_start = time.time()
        
        # Explicit memory cleanup to free unified memory
        del segments, info
        del batched_model
        del model
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        cleanup_end = time.time()
        cleanup_time = cleanup_end - cleanup_start
        log_with_timestamp(f"[{model_size}] ✅ MEMORY CLEANUP - END (Cleanup time: {cleanup_time:.2f}s)")
        log_with_timestamp(f"[{model_size}] Model {model_size} testing completed successfully")
        
    except Exception as e:
        log_with_timestamp(f"[{model_size}] ❌ ERROR: {str(e)}")
        results.append({
            'model_size': model_size,
            'processing_time': None,
            'model_load_time': None,
            'wer': None,
            'cer': None,
            'unified_memory_gb': None,
            'transcript': f"Error: {str(e)}",
            'load_start_time': None,
            'load_end_time': None,
            'transcription_start_time': None,
            'transcription_end_time': None
        })
        
        # Clean up even on error
        try:
            if 'batched_model' in locals():
                del batched_model
            if 'model' in locals():
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_with_timestamp(f"[{model_size}] Emergency cleanup completed")
        except:
            pass

log_with_timestamp("\n=== ALL MODEL TESTING COMPLETED ===")

# Create DataFrame for easier analysis
df = pd.DataFrame(results)

# Filter out failed models for plotting
df_success = df.dropna(subset=['processing_time', 'wer'])

log_with_timestamp(f"Successfully tested {len(df_success)} out of {len(model_sizes)} models")

if len(df_success) > 0:
    # Create comparison charts (updated for unified memory)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
    
    # Chart 1: Processing Time Comparison
    ax1.bar(df_success['model_size'], df_success['processing_time'], color='skyblue')
    ax1.set_title('Processing Time by Model Size')
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df_success['processing_time']):
        ax1.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')
    
    # Chart 2: WER Comparison
    ax2.bar(df_success['model_size'], df_success['wer'] * 100, color='lightcoral')
    ax2.set_title('Word Error Rate (WER) by Model Size')
    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('WER (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df_success['wer']):
        ax2.text(i, v * 100 + 0.5, f'{v*100:.2f}%', ha='center', va='bottom')
    
    # Chart 3: Unified Memory Usage (Jetson Orin)
    ax3.bar(df_success['model_size'], df_success['unified_memory_gb'], color='orange')
    ax3.set_title('Unified Memory Usage by Model Size (Jetson Orin)')
    ax3.set_xlabel('Model Size')
    ax3.set_ylabel('Unified Memory (GB)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df_success['unified_memory_gb']):
        ax3.text(i, v + 0.1, f'{v:.2f}GB', ha='center', va='bottom')
    
    # Chart 4: CER Comparison
    ax4.bar(df_success['model_size'], df_success['cer'] * 100, color='purple')
    ax4.set_title('Character Error Rate (CER) by Model Size')
    ax4.set_xlabel('Model Size')
    ax4.set_ylabel('CER (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df_success['cer']):
        ax4.text(i, v * 100 + 0.5, f'{v*100:.2f}%', ha='center', va='bottom')
    
    # Chart 5: Time vs WER Scatter Plot
    ax5.scatter(df_success['processing_time'], df_success['wer'] * 100, 
                c='green', alpha=0.7, s=100)
    for i, txt in enumerate(df_success['model_size']):
        ax5.annotate(txt, (df_success['processing_time'].iloc[i], 
                          df_success['wer'].iloc[i] * 100), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax5.set_title('Processing Time vs WER')
    ax5.set_xlabel('Processing Time (seconds)')
    ax5.set_ylabel('WER (%)')
    ax5.grid(True, alpha=0.3)
    
    # Chart 6: Unified Memory vs Accuracy Trade-off (Jetson Orin)
    ax6.scatter(df_success['unified_memory_gb'], df_success['wer'] * 100, 
                c='red', alpha=0.7, s=100)
    for i, txt in enumerate(df_success['model_size']):
        ax6.annotate(txt, (df_success['unified_memory_gb'].iloc[i], 
                          df_success['wer'].iloc[i] * 100), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax6.set_title('Unified Memory vs WER Trade-off (Jetson Orin)')
    ax6.set_xlabel('Unified Memory (GB)')
    ax6.set_ylabel('WER (%)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('whisper_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table with unified memory
    log_with_timestamp("\n" + "="*90)
    log_with_timestamp("SUMMARY TABLE (Jetson Orin Unified Memory)")
    log_with_timestamp("="*90)
    log_with_timestamp(f"{'Model':<20} {'Load (s)':<10} {'Trans (s)':<12} {'WER (%)':<12} {'CER (%)':<12} {'Unified Mem (GB)':<18}")
    log_with_timestamp("-" * 90)
    
    for _, row in df_success.iterrows():
        load_time_str = f"{row['model_load_time']:.2f}" if row['model_load_time'] is not None else "N/A"
        log_with_timestamp(f"{row['model_size']:<20} {load_time_str:<10} {row['processing_time']:<12.2f} "
              f"{row['wer']*100:<12.2f} {row['cer']*100:<12.2f} "
              f"{row['unified_memory_gb']:<18.2f}")
    
    # Find best models including unified memory metrics
    if len(df_success) > 0:
        best_speed = df_success.loc[df_success['processing_time'].idxmin()]
        best_wer = df_success.loc[df_success['wer'].idxmin()]
        best_cer = df_success.loc[df_success['cer'].idxmin()]
        lowest_memory = df_success.loc[df_success['unified_memory_gb'].idxmin()]
        
        log_with_timestamp("\n" + "="*90)
        log_with_timestamp("BEST PERFORMING MODELS (Jetson Orin)")
        log_with_timestamp("="*90)
        log_with_timestamp(f"Fastest: {best_speed['model_size']} ({best_speed['processing_time']:.2f}s)")
        log_with_timestamp(f"Lowest WER: {best_wer['model_size']} ({best_wer['wer']*100:.2f}%)")
        log_with_timestamp(f"Lowest CER: {best_cer['model_size']} ({best_cer['cer']*100:.2f}%)")
        log_with_timestamp(f"Lowest Unified Memory: {lowest_memory['model_size']} ({lowest_memory['unified_memory_gb']:.2f}GB)")
        
        # Calculate efficiency score (lower is better)
        df_success['efficiency_score'] = df_success['processing_time'] * df_success['wer']
        best_efficiency = df_success.loc[df_success['efficiency_score'].idxmin()]
        log_with_timestamp(f"Best Overall (Time × WER): {best_efficiency['model_size']} "
              f"(Score: {best_efficiency['efficiency_score']:.4f})")
        
        # Memory efficiency score for unified memory (WER / Unified Memory - lower is better)
        df_success['memory_efficiency'] = df_success['wer'] / df_success['unified_memory_gb']
        best_memory_efficiency = df_success.loc[df_success['memory_efficiency'].idxmin()]
        log_with_timestamp(f"Best Memory Efficiency: {best_memory_efficiency['model_size']} "
              f"(WER/GB: {best_memory_efficiency['memory_efficiency']:.4f})")
else:
    log_with_timestamp("No models were successfully tested. Please check your setup.")

# Save detailed results to CSV with timing information
df.to_csv('whisper_model_results.csv', index=False)
log_with_timestamp(f"\nDetailed results saved to 'whisper_model_results.csv'")
log_with_timestamp(f"Timing log saved to '{timing_log_file}'")
log_with_timestamp(f"Comparison chart saved to 'whisper_model_comparison.png'")
log_with_timestamp("\n=== TESTING SESSION COMPLETE ===")
log_with_timestamp(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_with_timestamp(f"Total duration: {time.time() - time.time():.2f} seconds")  # This will be calculated properly in full script

