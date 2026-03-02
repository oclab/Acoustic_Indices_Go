package main

import (
	"math"
)

// IndexResult holds the computed values for one acoustic index.
type IndexResult struct {
	MainValue      *float64
	TemporalValues []float64
	Extra          map[string]float64
}

// computeACI computes the Acoustic Complexity Index.
// Reference: Pieretti N, Farina A, Morri FD (2011).
// jBinSamples: temporal window size in spectrogram frames.
// Returns (mainValue, temporalValues).
func computeACI(spectro [][]float64, jBinSamples int) (float64, []float64) {
	numFreqs := len(spectro)
	if numFreqs == 0 {
		return 0, nil
	}
	numTimes := len(spectro[0])

	var aciValues []float64
	for start := 0; start <= numTimes-10-jBinSamples; start += jBinSamples {
		end := start + jBinSamples
		if end > numTimes {
			end = numTimes
		}
		segLen := end - start
		if segLen < 2 {
			continue
		}

		var segACI float64
		for f := 0; f < numFreqs; f++ {
			sumDiff := 0.0
			sumVal := 0.0
			for t := start; t < end-1; t++ {
				sumDiff += math.Abs(spectro[f][t+1] - spectro[f][t])
			}
			for t := start; t < end; t++ {
				sumVal += spectro[f][t]
			}
			if sumVal > 0 {
				segACI += sumDiff / sumVal
			}
		}
		aciValues = append(aciValues, segACI)
	}

	mainValue := 0.0
	for _, v := range aciValues {
		mainValue += v
	}
	return mainValue, aciValues
}

// computeBI computes the Bioacoustic Index.
// Reference: Boelman et al. (2007).
func computeBI(spectro [][]float64, frequencies []float64, minFreq, maxFreq float64) float64 {
	if len(spectro) == 0 || len(frequencies) == 0 {
		return 0
	}

	minBin := argminAbsDiff(frequencies, minFreq) - 1 // match Python: min_freq_bin - 1
	if minBin < 0 {
		minBin = 0
	}
	maxBin := argminAbsDiff(frequencies, maxFreq)

	// Find global max across spectrogram
	maxVal := 0.0
	for _, row := range spectro {
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
	}
	if maxVal == 0 {
		return 0
	}

	numTimes := len(spectro[0])
	numFreqs := len(spectro)

	// dB spectrogram
	specDB := make([][]float64, numFreqs)
	for f := range specDB {
		specDB[f] = make([]float64, numTimes)
		for t := range specDB[f] {
			specDB[f][t] = 20.0 * math.Log10(spectro[f][t]/maxVal)
		}
	}

	// Mean spectrum using log-domain mean: 10*log10(mean(10^(x/10)))
	meanSpec := make([]float64, numFreqs)
	for f := 0; f < numFreqs; f++ {
		sumPow := 0.0
		for t := 0; t < numTimes; t++ {
			sumPow += math.Pow(10.0, specDB[f][t]/10.0)
		}
		meanSpec[f] = 10.0 * math.Log10(sumPow/float64(numTimes))
	}

	// Segment and normalize
	segment := meanSpec[minBin:maxBin]
	if len(segment) == 0 {
		return 0
	}
	minSeg := segment[0]
	for _, v := range segment {
		if v < minSeg {
			minSeg = v
		}
	}

	freqRes := frequencies[1] - frequencies[0]
	if freqRes == 0 {
		return 0
	}

	area := 0.0
	for _, v := range segment {
		area += (v - minSeg) / freqRes
	}
	return area
}

// computeSH computes the Spectral Entropy of Shannon.
// Ported from the seewave R package.
func computeSH(spectro [][]float64) float64 {
	numFreqs := len(spectro)
	if numFreqs == 0 {
		return 0
	}

	// Sum along time axis for each frequency bin
	spec := make([]float64, numFreqs)
	for f := 0; f < numFreqs; f++ {
		for _, v := range spectro[f] {
			spec[f] += v
		}
	}

	// Normalize
	total := 0.0
	for _, v := range spec {
		total += v
	}
	if total == 0 {
		return 0
	}
	for i := range spec {
		spec[i] /= total
	}

	// Shannon entropy
	entropy := 0.0
	for _, v := range spec {
		if v > 0 {
			entropy -= v * math.Log2(v)
		}
	}
	return entropy / math.Log2(float64(numFreqs))
}

// computeTH computes the Temporal Entropy of Shannon using the Hilbert envelope.
// Ported from the seewave R package.
func computeTH(sig []float64) float64 {
	env := hilbertEnvelope(sig)

	total := 0.0
	for _, v := range env {
		total += v
	}
	if total == 0 {
		return 0
	}

	N := len(env)
	entropy := 0.0
	for _, v := range env {
		p := v / total
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy / math.Log2(float64(N))
}

// computeNDSI computes the Normalized Difference Sound Index.
// Reference: Kasten et al. (2012).
func computeNDSI(sig []float64, sr, windowLength int, anthrophony, biophony [2]float64) float64 {
	freqs, pxx := welchPSD(sig, sr, windowLength)

	// avgpow = pxx * freqResolution (rectangle approximation of PSD integral)
	freqRes := freqs[1]
	avgpow := make([]float64, len(pxx))
	for i, v := range pxx {
		avgpow[i] = v * freqRes
	}

	minAnthro := argminAbsDiff(freqs, anthrophony[0])
	maxAnthro := argminAbsDiff(freqs, anthrophony[1])
	minBio := argminAbsDiff(freqs, biophony[0])
	maxBio := argminAbsDiff(freqs, biophony[1])

	anthro := 0.0
	for _, v := range avgpow[minAnthro:maxAnthro] {
		anthro += v
	}
	bio := 0.0
	for _, v := range avgpow[minBio:maxBio] {
		bio += v
	}

	if bio+anthro == 0 {
		return 0
	}
	return (bio - anthro) / (bio + anthro)
}

// gini computes the Gini index of a slice of values.
func gini(values []float64) float64 {
	n := len(values)
	if n == 0 {
		return 0
	}
	y := make([]float64, n)
	copy(y, values)
	// Sort y ascending
	sortFloat64(y)

	sumTotal := 0.0
	for _, v := range y {
		sumTotal += v
	}
	if sumTotal == 0 {
		return 0
	}

	G := 0.0
	for i, v := range y {
		G += float64(i+1) * v
	}
	G = 2*G/sumTotal - float64(n+1)
	return G / float64(n)
}

// computeAEI computes the Acoustic Evenness Index.
// Reference: Villanueva-Rivera et al. (2011).
func computeAEI(spectro [][]float64, freqBandHz float64, maxFreq float64, dbThreshold float64, freqStep float64) float64 {
	if len(spectro) == 0 {
		return 0
	}
	numFreqs := len(spectro)
	numTimes := len(spectro[0])

	// Global max for dB normalization
	maxVal := 0.0
	for _, row := range spectro {
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
	}
	if maxVal == 0 {
		return 0
	}

	// dB spectrogram
	specDB := make([][]float64, numFreqs)
	for f := range specDB {
		specDB[f] = make([]float64, numTimes)
		for t := range specDB[f] {
			specDB[f][t] = 20.0 * math.Log10(spectro[f][t]/maxVal)
		}
	}

	bandBin := freqStep / freqBandHz
	numBands := int(maxFreq / freqStep)
	values := make([]float64, numBands)

	for k := 0; k < numBands; k++ {
		startBin := int(float64(k) * bandBin)
		endBin := int(float64(k)*bandBin + bandBin)
		if endBin > numFreqs {
			endBin = numFreqs
		}
		if startBin >= endBin {
			continue
		}
		total := 0
		above := 0
		for f := startBin; f < endBin; f++ {
			for t := 0; t < numTimes; t++ {
				total++
				if specDB[f][t] > dbThreshold {
					above++
				}
			}
		}
		if total > 0 {
			values[k] = float64(above) / float64(total)
		}
	}

	return gini(values)
}

// computeADI computes the Acoustic Diversity Index.
// Reference: Villanueva-Rivera et al. (2011).
func computeADI(spectro [][]float64, freqBandHz float64, maxFreq float64, dbThreshold float64, freqStep float64) float64 {
	if len(spectro) == 0 {
		return 0
	}
	numFreqs := len(spectro)
	numTimes := len(spectro[0])

	// Global max for dB normalization
	maxVal := 0.0
	for _, row := range spectro {
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
	}
	if maxVal == 0 {
		return 0
	}

	// dB spectrogram
	specDB := make([][]float64, numFreqs)
	for f := range specDB {
		specDB[f] = make([]float64, numTimes)
		for t := range specDB[f] {
			specDB[f][t] = 20.0 * math.Log10(spectro[f][t]/maxVal)
		}
	}

	bandBin := freqStep / freqBandHz
	numBands := int(maxFreq / freqStep)
	values := make([]float64, 0, numBands)

	for k := 0; k < numBands; k++ {
		startBin := int(float64(k) * bandBin)
		endBin := int(float64(k)*bandBin + bandBin)
		if endBin > numFreqs {
			endBin = numFreqs
		}
		if startBin >= endBin {
			continue
		}
		total := 0
		above := 0
		for f := startBin; f < endBin; f++ {
			for t := 0; t < numTimes; t++ {
				total++
				if specDB[f][t] > dbThreshold {
					above++
				}
			}
		}
		if total > 0 {
			v := float64(above) / float64(total)
			if v != 0 {
				values = append(values, v)
			}
		}
	}

	if len(values) == 0 {
		return 0
	}

	sumVal := 0.0
	for _, v := range values {
		sumVal += v
	}
	if sumVal == 0 {
		return 0
	}

	adi := 0.0
	for _, v := range values {
		p := v / sumVal
		adi -= p * math.Log(p)
	}
	return adi
}

// computeZCR computes the Zero Crossing Rate.
func computeZCR(sigInt []int, windowLength, windowHop int) []float64 {
	var result []float64
	for i := 0; i+windowLength <= len(sigInt); i += windowHop {
		frame := sigInt[i : i+windowLength]
		crossings := 0
		for j := 0; j < len(frame)-1; j++ {
			if (frame[j] >= 0) != (frame[j+1] >= 0) {
				crossings++
			}
		}
		result = append(result, float64(crossings)/float64(windowLength))
	}
	return result
}

// computeRMSEnergy computes the RMS short-time energy.
// If useFloat is true, use the floating-point signal; otherwise use integer.
func computeRMSEnergy(sigFloat []float64, sigInt []int, windowLength, windowHop int, useFloat bool) []float64 {
	var result []float64
	n := len(sigFloat)
	if !useFloat {
		n = len(sigInt)
	}
	for i := 0; i+windowLength <= n; i += windowHop {
		sumSq := 0.0
		for j := 0; j < windowLength; j++ {
			var v float64
			if useFloat {
				v = sigFloat[i+j]
			} else {
				v = float64(sigInt[i+j])
			}
			sumSq += v * v
		}
		result = append(result, math.Sqrt(sumSq/float64(windowLength)))
	}
	return result
}

// computeSpectralCentroid computes the spectral centroid for each time frame.
// spectro[freq][time]; frequencies[freq].
func computeSpectralCentroid(spectro [][]float64, frequencies []float64) []float64 {
	if len(spectro) == 0 {
		return nil
	}
	numTimes := len(spectro[0])
	result := make([]float64, numTimes)
	for t := 0; t < numTimes; t++ {
		sumMagFreq := 0.0
		sumMag := 0.0
		for f := range spectro {
			mag := spectro[f][t]
			sumMagFreq += mag * frequencies[f]
			sumMag += mag
		}
		if sumMag > 0 {
			result[t] = sumMagFreq / sumMag
		}
	}
	return result
}

// WaveSNRResult holds the output of Wave SNR computation.
type WaveSNRResult struct {
	SNR                float64
	AcousticActivity   float64
	CountAcousticEvents int
	AverageDuration    float64
}

// computeWaveSNR computes Signal-to-Noise Ratio indices from the waveform.
// Reference: Towsey (2013).
func computeWaveSNR(
	sigFloat []float64,
	duration float64,
	frameLengthE int,
	minDB float64,
	windowSmoothingE int,
	activityThresholdDB float64,
	histNumberBins int,
	dbRange float64,
	N float64,
) WaveSNRResult {
	halfSmooth := windowSmoothingE / 2

	// Compute wave envelope in dB
	var waveEnv []float64
	for i := 0; i+frameLengthE <= len(sigFloat); i += frameLengthE {
		frame := sigFloat[i : i+frameLengthE]
		maxAbs := 0.0
		for _, v := range frame {
			a := math.Abs(v)
			if a > maxAbs {
				maxAbs = a
			}
		}
		if maxAbs > 0 {
			waveEnv = append(waveEnv, 20.0*math.Log10(maxAbs))
		} else {
			waveEnv = append(waveEnv, minDB)
		}
	}

	if len(waveEnv) == 0 {
		return WaveSNRResult{}
	}

	minEnv := waveEnv[0]
	for _, v := range waveEnv {
		if v < minEnv {
			minEnv = v
		}
	}
	minimum := math.Max(minEnv, minDB)

	// Build histogram
	hist := make([]float64, histNumberBins)
	for _, v := range waveEnv {
		bin := int((v - minimum) / dbRange * float64(histNumberBins))
		if bin < 0 {
			bin = 0
		}
		if bin >= histNumberBins {
			bin = histNumberBins - 1
		}
		hist[bin]++
	}

	// Smooth histogram
	histSmooth := make([]float64, histNumberBins)
	for i := halfSmooth; i < histNumberBins-halfSmooth; i++ {
		sum := 0.0
		for j := i - halfSmooth; j < i+halfSmooth; j++ {
			sum += hist[j]
		}
		histSmooth[i] = sum / float64(2*halfSmooth)
	}

	// Find modal intensity
	modalIntensity := 0
	for i, v := range histSmooth {
		if v > histSmooth[modalIntensity] {
			modalIntensity = i
		}
	}

	// Background noise level
	binEdge := func(bin int) float64 {
		return minimum + float64(bin)/float64(histNumberBins)*dbRange
	}

	var backgroundNoise float64
	if N > 0 {
		totalCount := 0.0
		for _, v := range histSmooth {
			totalCount += v
		}
		countThresh := 0.68 * totalCount
		count := histSmooth[modalIntensity]
		indexBin := 1
		for count < countThresh {
			if modalIntensity+indexBin < histNumberBins {
				count += histSmooth[modalIntensity+indexBin]
			}
			if modalIntensity-indexBin >= 0 {
				count += histSmooth[modalIntensity-indexBin]
			}
			indexBin++
		}
		thresh := modalIntensity + int(N)*indexBin
		if thresh >= histNumberBins {
			thresh = histNumberBins - 1
		}
		backgroundNoise = binEdge(thresh)
	} else {
		backgroundNoise = binEdge(modalIntensity)
	}

	// Compute SN for each frame
	sn := make([]float64, len(waveEnv))
	for i, v := range waveEnv {
		sn[i] = v - backgroundNoise - activityThresholdDB
	}

	maxEnv := waveEnv[0]
	for _, v := range waveEnv {
		if v > maxEnv {
			maxEnv = v
		}
	}
	snr := maxEnv - backgroundNoise

	// Acoustic activity
	activeCount := 0
	for _, v := range sn {
		if v > 0 {
			activeCount++
		}
	}
	acousticActivity := float64(activeCount) / float64(len(sn))

	// Acoustic events
	countEvents := 0
	totalDuration := 0.0

	inEvent := false
	eventStart := 0
	for i, v := range sn {
		if !inEvent && v > 0 {
			inEvent = true
			eventStart = i
		} else if inEvent && v <= 0 {
			inEvent = false
			dur := float64(i-eventStart) * duration / float64(len(sn))
			totalDuration += dur
			countEvents++
		}
	}

	avgDuration := 0.0
	if countEvents > 0 {
		avgDuration = totalDuration / float64(countEvents)
	}

	return WaveSNRResult{
		SNR:                snr,
		AcousticActivity:   acousticActivity,
		CountAcousticEvents: countEvents,
		AverageDuration:    avgDuration,
	}
}

// computeNBPeaks counts the number of major frequency peaks in the mean spectrum.
// Reference: Gasc et al. (2013).
func computeNBPeaks(spectro [][]float64, frequencies []float64, freqband float64, normalization bool, slopeLeft, slopeRight float64) int {
	numFreqs := len(spectro)
	if numFreqs == 0 {
		return 0
	}

	// Compute mean spectrum
	meanSpec := make([]float64, numFreqs)
	for f := 0; f < numFreqs; f++ {
		sum := 0.0
		for _, v := range spectro[f] {
			sum += v
		}
		meanSpec[f] = sum / float64(len(spectro[f]))
	}

	if normalization {
		maxVal := 0.0
		for _, v := range meanSpec {
			if v > maxVal {
				maxVal = v
			}
		}
		if maxVal > 0 {
			for i := range meanSpec {
				meanSpec[i] /= maxVal
			}
		}
	}

	// Find peaks using slope criterion
	var peakIndices []int
	for i := 1; i < numFreqs-1; i++ {
		if meanSpec[i] > meanSpec[i-1]+slopeLeft && meanSpec[i] > meanSpec[i+1]+slopeRight {
			peakIndices = append(peakIndices, i)
		}
	}

	// Find number of bins corresponding to freqband
	nbBin := 0
	for i, f := range frequencies {
		if f > freqband {
			nbBin = i
			break
		}
	}
	if nbBin == 0 {
		nbBin = 1
	}

	// Remove peaks that are too close together (keep highest)
	changed := true
	for changed {
		changed = false
		for i := 0; i < len(peakIndices); i++ {
			for j := i + 1; j < len(peakIndices); j++ {
				if peakIndices[j]-peakIndices[i] < nbBin {
					// Keep the one with higher amplitude
					if meanSpec[peakIndices[i]] >= meanSpec[peakIndices[j]] {
						peakIndices = append(peakIndices[:j], peakIndices[j+1:]...)
					} else {
						peakIndices = append(peakIndices[:i], peakIndices[i+1:]...)
					}
					changed = true
					break
				}
			}
			if changed {
				break
			}
		}
	}

	return len(peakIndices)
}

// argminAbsDiff returns the index i where |values[i] - target| is minimized.
func argminAbsDiff(values []float64, target float64) int {
	if len(values) == 0 {
		return 0
	}
	best := 0
	bestDiff := math.Abs(values[0] - target)
	for i := 1; i < len(values); i++ {
		d := math.Abs(values[i] - target)
		if d < bestDiff {
			bestDiff = d
			best = i
		}
	}
	return best
}

// sortFloat64 sorts a float64 slice in ascending order (insertion sort for small slices).
func sortFloat64(v []float64) {
	for i := 1; i < len(v); i++ {
		key := v[i]
		j := i - 1
		for j >= 0 && v[j] > key {
			v[j+1] = v[j]
			j--
		}
		v[j+1] = key
	}
}
