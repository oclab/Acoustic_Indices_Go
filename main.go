package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// Config holds the YAML configuration for filtering and indices.
type Config struct {
	Filtering *FilteringConfig         `yaml:"Filtering"`
	Indices   map[string]IndexConfig   `yaml:"Indices"`
}

// FilteringConfig holds pre-processing filter settings.
type FilteringConfig struct {
	Type      string  `yaml:"type"`
	Order     int     `yaml:"order"`
	Frequency float64 `yaml:"frequency"`
}

// IndexConfig holds per-index configuration.
type IndexConfig struct {
	Function           string                 `yaml:"function"`
	Spectro            map[string]interface{} `yaml:"spectro"`
	Arguments          map[string]interface{} `yaml:"arguments"`
	RemoveNoiseInSpectro map[string]interface{} `yaml:"remove_noiseInSpectro"`
}

// FileResult holds all computed index values for one audio file.
type FileResult struct {
	FileName string
	Values   map[string]float64
}

func main() {
	inputDir := flag.String("input", "audio_files", "directory containing WAV files")
	outputCSV := flag.String("output", "acoustic_indices.csv", "output CSV file path")
	workers := flag.Int("workers", runtime.NumCPU(), "number of parallel goroutines")
	configFile := flag.String("config", "yaml/config_014_butter.yaml", "YAML configuration file")
	flag.Parse()

	fmt.Printf("Acoustic Indices Extractor\n")
	fmt.Printf("  Input directory : %s\n", *inputDir)
	fmt.Printf("  Output CSV      : %s\n", *outputCSV)
	fmt.Printf("  Workers         : %d\n", *workers)
	fmt.Printf("  Config file     : %s\n\n", *configFile)

	// Load YAML config
	cfg, err := loadConfig(*configFile)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Collect WAV files
	wavFiles, err := findWAVFiles(*inputDir)
	if err != nil {
		log.Fatalf("Failed to scan directory: %v", err)
	}
	if len(wavFiles) == 0 {
		log.Fatalf("No WAV files found in %s", *inputDir)
	}
	fmt.Printf("Found %d WAV file(s)\n\n", len(wavFiles))

	// Determine CSV column order from config
	indexNames := orderedIndexNames(cfg)

	// Process files in parallel
	results := make([]FileResult, len(wavFiles))
	jobs := make(chan int, len(wavFiles))
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i := 0; i < len(wavFiles); i++ {
		jobs <- i
	}
	close(jobs)

	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				path := wavFiles[idx]
				result, err := processFile(path, cfg, indexNames)
				if err != nil {
					mu.Lock()
					log.Printf("ERROR processing %s: %v\n", path, err)
					mu.Unlock()
					continue
				}
				mu.Lock()
				fmt.Printf("  Processed: %s\n", result.FileName)
				results[idx] = result
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	// Write CSV
	if err := writeCSV(*outputCSV, indexNames, results); err != nil {
		log.Fatalf("Failed to write CSV: %v", err)
	}
	fmt.Printf("\nResults written to %s\n", *outputCSV)
}

// loadConfig parses the YAML configuration file.
func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// findWAVFiles walks a directory and returns all .wav file paths.
func findWAVFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".wav") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

// orderedIndexNames returns the index names in config order (map order from YAML).
func orderedIndexNames(cfg *Config) []string {
	var names []string
	for name := range cfg.Indices {
		names = append(names, name)
	}
	return names
}

// processFile loads one WAV file and computes all configured acoustic indices.
func processFile(filePath string, cfg *Config, indexNames []string) (FileResult, error) {
	af, err := LoadAudioFile(filePath)
	if err != nil {
		return FileResult{}, err
	}

	sig := af.SigFloat
	sigInt := af.SigInt

	// Optional pre-processing: high-pass filtering
	if cfg.Filtering != nil && cfg.Filtering.Type == "butterworth" {
		sig = ApplyButterworthHP(sig, af.SR, cfg.Filtering.Order, cfg.Filtering.Frequency)
	} else if cfg.Filtering != nil {
		log.Printf("Warning: filtering type '%s' not supported; skipping\n", cfg.Filtering.Type)
	}

	values := make(map[string]float64)

	for _, name := range indexNames {
		icfg, ok := cfg.Indices[name]
		if !ok {
			continue
		}

		switch name {
		case "Acoustic_Complexity_Index":
			spectro, _ := makeSpectro(sig, af.SR, icfg.Spectro)
			if spectro == nil {
				break
			}
			windowHop := getIntParam(icfg.Spectro, "windowHop", 512)
			jBinSecs := getFloatParam(icfg.Arguments, "j_bin", 5)
			jBinSamples := int(jBinSecs * float64(af.SR) / float64(windowHop))
			main, _ := computeACI(spectro, jBinSamples)
			values[name+"__main_value"] = main

		case "Acoustic_Diversity_Index":
			freqBandHz, spectro := makeADISpectro(sig, af.SR, icfg.Arguments)
			maxFreq := getFloatParam(icfg.Arguments, "max_freq", 10000)
			dbThresh := getFloatParam(icfg.Arguments, "db_threshold", -50)
			freqStep := getFloatParam(icfg.Arguments, "freq_step", 1000)
			values[name+"__main_value"] = computeADI(spectro, freqBandHz, maxFreq, dbThresh, freqStep)

		case "Acoustic_Evenness_Index":
			freqBandHz, spectro := makeADISpectro(sig, af.SR, icfg.Arguments)
			maxFreq := getFloatParam(icfg.Arguments, "max_freq", 10000)
			dbThresh := getFloatParam(icfg.Arguments, "db_threshold", -50)
			freqStep := getFloatParam(icfg.Arguments, "freq_step", 1000)
			values[name+"__main_value"] = computeAEI(spectro, freqBandHz, maxFreq, dbThresh, freqStep)

		case "Bio_acoustic_Index":
			spectro, frequencies := makeSpectro(sig, af.SR, icfg.Spectro)
			if spectro == nil {
				break
			}
			minFreq := getFloatParam(icfg.Arguments, "min_freq", 2000)
			maxFreq := getFloatParam(icfg.Arguments, "max_freq", 8000)
			values[name+"__main_value"] = computeBI(spectro, frequencies, minFreq, maxFreq)

		case "Normalized_Difference_Sound_Index":
			wl := getIntParam(icfg.Arguments, "windowLength", 1024)
			anthro := getFloatPairParam(icfg.Arguments, "anthrophony", 1000, 2000)
			bio := getFloatPairParam(icfg.Arguments, "biophony", 2000, 11000)
			values[name+"__main_value"] = computeNDSI(sig, af.SR, wl, anthro, bio)

		case "RMS_energy":
			wl := getIntParam(icfg.Arguments, "windowLength", 512)
			wh := getIntParam(icfg.Arguments, "windowHop", 256)
			useInt := getBoolParam(icfg.Arguments, "integer", false)
			temporal := computeRMSEnergy(sig, sigInt, wl, wh, !useInt)
			avg, mn, mx, sd := statsFloat64(temporal)
			values[name+"__mean"] = avg
			values[name+"__min"] = mn
			values[name+"__max"] = mx
			values[name+"__std"] = sd

		case "Spectral_centroid":
			spectro, frequencies := makeSpectro(sig, af.SR, icfg.Spectro)
			if spectro == nil {
				break
			}
			temporal := computeSpectralCentroid(spectro, frequencies)
			avg, mn, mx, sd := statsFloat64(temporal)
			values[name+"__mean"] = avg
			values[name+"__min"] = mn
			values[name+"__max"] = mx
			values[name+"__std"] = sd

		case "Spectral_Entropy":
			spectro, _ := makeSpectro(sig, af.SR, icfg.Spectro)
			if spectro == nil {
				break
			}
			values[name+"__main_value"] = computeSH(spectro)

		case "Temporal_Entropy":
			useInt := getBoolParam(icfg.Arguments, "integer", true)
			var input []float64
			if useInt {
				input = make([]float64, len(sigInt))
				for i, v := range sigInt {
					input[i] = float64(v)
				}
			} else {
				input = sig
			}
			values[name+"__main_value"] = computeTH(input)

		case "ZCR":
			wl := getIntParam(icfg.Arguments, "windowLength", 512)
			wh := getIntParam(icfg.Arguments, "windowHop", 256)
			temporal := computeZCR(sigInt, wl, wh)
			avg, mn, mx, sd := statsFloat64(temporal)
			values[name+"__mean"] = avg
			values[name+"__min"] = mn
			values[name+"__max"] = mx
			values[name+"__std"] = sd

		case "Wave_SNR":
			fl := getIntParam(icfg.Arguments, "frame_length_e", 512)
			minDB := getFloatParam(icfg.Arguments, "min_DB", -60)
			ws := getIntParam(icfg.Arguments, "window_smoothing_e", 5)
			at := getFloatParam(icfg.Arguments, "activity_threshold_dB", 3)
			hn := getIntParam(icfg.Arguments, "hist_number_bins", 100)
			dr := getFloatParam(icfg.Arguments, "dB_range", 10)
			res := computeWaveSNR(sig, af.Duration, fl, minDB, ws, at, hn, dr, 0)
			values[name+"__SNR"] = res.SNR
			values[name+"__Acoustic_activity"] = res.AcousticActivity
			values[name+"__Count_acoustic_events"] = float64(res.CountAcousticEvents)
			values[name+"__Average_duration"] = res.AverageDuration

		case "NB_peaks":
			spectro, frequencies := makeSpectro(sig, af.SR, icfg.Spectro)
			if spectro == nil {
				break
			}
			fb := getFloatParam(icfg.Arguments, "freqband", 200)
			norm := getBoolParam(icfg.Arguments, "normalization", true)
			slopes := getFloatSliceParam(icfg.Arguments, "slopes", []float64{0.01, 0.01})
			values[name+"__main_value"] = float64(computeNBPeaks(spectro, frequencies, fb, norm, slopes[0], slopes[1]))

		case "Acoustic_Diversity_Index_NR":
			freqBandHz, spectro := makeADISpectro(sig, af.SR, icfg.Arguments)
			spectroNR := applyNoiseRemoval(spectro, icfg.RemoveNoiseInSpectro)
			maxFreq := getFloatParam(icfg.Arguments, "max_freq", 10000)
			dbThresh := getFloatParam(icfg.Arguments, "db_threshold", -50)
			freqStep := getFloatParam(icfg.Arguments, "freq_step", 1000)
			values[name+"__main_value"] = computeADI(spectroNR, freqBandHz, maxFreq, dbThresh, freqStep)

		case "Acoustic_Evenness_Index_NR":
			freqBandHz, spectro := makeADISpectro(sig, af.SR, icfg.Arguments)
			spectroNR := applyNoiseRemoval(spectro, icfg.RemoveNoiseInSpectro)
			maxFreq := getFloatParam(icfg.Arguments, "max_freq", 10000)
			dbThresh := getFloatParam(icfg.Arguments, "db_threshold", -50)
			freqStep := getFloatParam(icfg.Arguments, "freq_step", 1000)
			values[name+"__main_value"] = computeAEI(spectroNR, freqBandHz, maxFreq, dbThresh, freqStep)

		case "Bio_acoustic_Index_NR":
			spectro, frequencies := makeSpectro(sig, af.SR, icfg.Spectro)
			if spectro == nil {
				break
			}
			spectroNR := applyNoiseRemoval(spectro, icfg.RemoveNoiseInSpectro)
			minFreq := getFloatParam(icfg.Arguments, "min_freq", 2000)
			maxFreq := getFloatParam(icfg.Arguments, "max_freq", 8000)
			values[name+"__main_value"] = computeBI(spectroNR, frequencies, minFreq, maxFreq)

		case "Spectral_Entropy_NR":
			spectro, _ := makeSpectro(sig, af.SR, icfg.Spectro)
			if spectro == nil {
				break
			}
			spectroNR := applyNoiseRemoval(spectro, icfg.RemoveNoiseInSpectro)
			values[name+"__main_value"] = computeSH(spectroNR)
		}
	}

	return FileResult{FileName: af.FileName, Values: values}, nil
}

// makeSpectro builds a spectrogram from YAML spectro config.
func makeSpectro(sig []float64, sr int, params map[string]interface{}) ([][]float64, []float64) {
	wl := getIntParam(params, "windowLength", 512)
	wh := getIntParam(params, "windowHop", 256)
	wt := getStringParam(params, "windowType", "hanning")
	scale := getBoolParam(params, "scale_audio", true)
	square := getBoolParam(params, "square", true)
	norm := getBoolParam(params, "normalized", false)

	input := sig
	if !scale {
		// Use unnormalized (the float signal IS already the normalized form,
		// and for scale_audio=False in Python the integer values were used;
		// our float signal is the equivalent normalized form, so we keep it)
		input = sig
	}
	return computeSpectrogram(input, sr, wl, wh, wt, square, norm)
}

// makeADISpectro builds the spectrogram for ADI/AEI (square=false, using freq_step/max_freq).
func makeADISpectro(sig []float64, sr int, args map[string]interface{}) (float64, [][]float64) {
	maxFreq := getFloatParam(args, "max_freq", 10000)
	freqStep := getFloatParam(args, "freq_step", 1000)
	freqBandHz := maxFreq / freqStep
	wl := int(float64(sr) / freqBandHz)
	spectro, _ := computeSpectrogram(sig, sr, wl, wl, "hanning", false, false)
	return freqBandHz, spectro
}

// applyNoiseRemoval applies noise removal to a spectrogram using YAML params.
func applyNoiseRemoval(spectro [][]float64, params map[string]interface{}) [][]float64 {
	if spectro == nil {
		return nil
	}
	histo := getIntParam(params, "histo_relative_size", 8)
	window := getIntParam(params, "window_smoothing", 5)
	return removeNoiseInSpectro(spectro, histo, window, 0.1, false)
}

// writeCSV writes all results to a CSV file.
func writeCSV(path string, indexNames []string, results []FileResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)

	// Build column order from first non-empty result
	var colKeys []string
	for _, r := range results {
		if len(r.Values) > 0 {
			colKeys = buildColumnKeys(indexNames, r.Values)
			break
		}
	}

	// Header
	header := append([]string{"filename"}, colKeys...)
	if err := w.Write(header); err != nil {
		return err
	}

	// Data rows
	for _, r := range results {
		if r.FileName == "" {
			continue
		}
		row := make([]string, len(header))
		row[0] = r.FileName
		for i, col := range colKeys {
			if v, ok := r.Values[col]; ok {
				row[i+1] = strconv.FormatFloat(v, 'f', 6, 64)
			}
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	w.Flush()
	return w.Error()
}

// buildColumnKeys returns CSV column names in a stable order derived from indexNames.
func buildColumnKeys(indexNames []string, sample map[string]float64) []string {
	// For each index, add its known sub-keys in a consistent order
	suffixOrder := map[string][]string{
		"Acoustic_Complexity_Index":          {"main_value"},
		"Acoustic_Diversity_Index":           {"main_value"},
		"Acoustic_Evenness_Index":            {"main_value"},
		"Bio_acoustic_Index":                 {"main_value"},
		"Normalized_Difference_Sound_Index":  {"main_value"},
		"RMS_energy":                         {"mean", "min", "max", "std"},
		"Spectral_centroid":                  {"mean", "min", "max", "std"},
		"Spectral_Entropy":                   {"main_value"},
		"Temporal_Entropy":                   {"main_value"},
		"ZCR":                                {"mean", "min", "max", "std"},
		"Wave_SNR":                           {"SNR", "Acoustic_activity", "Count_acoustic_events", "Average_duration"},
		"NB_peaks":                           {"main_value"},
		"Acoustic_Diversity_Index_NR":        {"main_value"},
		"Acoustic_Evenness_Index_NR":         {"main_value"},
		"Bio_acoustic_Index_NR":              {"main_value"},
		"Spectral_Entropy_NR":                {"main_value"},
	}

	var cols []string
	seen := make(map[string]bool)

	for _, name := range indexNames {
		if suffixes, ok := suffixOrder[name]; ok {
			for _, suf := range suffixes {
				col := name + "__" + suf
				if _, exists := sample[col]; exists && !seen[col] {
					cols = append(cols, col)
					seen[col] = true
				}
			}
		}
	}

	// Append any remaining keys from sample that weren't in our order map
	for k := range sample {
		if !seen[k] {
			cols = append(cols, k)
			seen[k] = true
		}
	}

	return cols
}

// statsFloat64 returns (mean, min, max, stddev) of a slice.
func statsFloat64(v []float64) (mean, min, max, std float64) {
	if len(v) == 0 {
		return
	}
	min = v[0]
	max = v[0]
	sum := 0.0
	for _, x := range v {
		sum += x
		if x < min {
			min = x
		}
		if x > max {
			max = x
		}
	}
	mean = sum / float64(len(v))
	varSum := 0.0
	for _, x := range v {
		d := x - mean
		varSum += d * d
	}
	if len(v) > 0 {
		std = math.Sqrt(varSum / float64(len(v)))
	}
	return
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// --- YAML parameter helpers ---

func getIntParam(m map[string]interface{}, key string, def int) int {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		switch x := v.(type) {
		case int:
			return x
		case float64:
			return int(x)
		}
	}
	return def
}

func getFloatParam(m map[string]interface{}, key string, def float64) float64 {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		switch x := v.(type) {
		case float64:
			return x
		case int:
			return float64(x)
		}
	}
	return def
}

func getBoolParam(m map[string]interface{}, key string, def bool) bool {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return def
}

func getStringParam(m map[string]interface{}, key string, def string) string {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return def
}

func getFloatPairParam(m map[string]interface{}, key string, def1, def2 float64) [2]float64 {
	if m == nil {
		return [2]float64{def1, def2}
	}
	if v, ok := m[key]; ok {
		if slice, ok := v.([]interface{}); ok && len(slice) == 2 {
			a := toFloat64(slice[0], def1)
			b := toFloat64(slice[1], def2)
			return [2]float64{a, b}
		}
	}
	return [2]float64{def1, def2}
}

func getFloatSliceParam(m map[string]interface{}, key string, def []float64) []float64 {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		if slice, ok := v.([]interface{}); ok {
			result := make([]float64, len(slice))
			for i, x := range slice {
				result[i] = toFloat64(x, 0)
			}
			return result
		}
	}
	return def
}

func toFloat64(v interface{}, def float64) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	}
	return def
}
