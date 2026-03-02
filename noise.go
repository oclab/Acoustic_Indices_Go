package main

import "math"

// removeNoiseInSpectro computes a noise-removed spectrogram.
// Reference: Towsey (2013) - modal intensity method.
// spectro[freq][time]; N: threshold std devs above modal intensity.
func removeNoiseInSpectro(spectro [][]float64, histoRelativeSize, windowSmoothing int, N float64, dB bool) [][]float64 {
	const lowValue = 1e-7
	halfSmooth := windowSmoothing / 2

	numFreqs := len(spectro)
	if numFreqs == 0 {
		return nil
	}
	numTimes := len(spectro[0])

	// Optionally convert to dB
	working := spectro
	if dB {
		working = make([][]float64, numFreqs)
		for f := range working {
			working[f] = make([]float64, numTimes)
			for t, v := range spectro[f] {
				if v > 0 {
					working[f][t] = 20.0 * math.Log10(v)
				}
			}
		}
	}

	// Compute background noise per frequency row
	backgroundNoise := make([]float64, numFreqs)
	for f := 0; f < numFreqs; f++ {
		row := working[f]
		histoSize := numTimes / histoRelativeSize
		if histoSize < 1 {
			histoSize = 1
		}

		minVal, maxVal := row[0], row[0]
		for _, v := range row {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		span := maxVal - minVal
		if span == 0 {
			backgroundNoise[f] = minVal
			continue
		}

		hist := make([]float64, histoSize)
		for _, v := range row {
			bin := int((v - minVal) / span * float64(histoSize))
			if bin >= histoSize {
				bin = histoSize - 1
			}
			hist[bin]++
		}

		// Smooth histogram
		histSmooth := make([]float64, histoSize)
		for i := halfSmooth; i < histoSize-halfSmooth; i++ {
			sum := 0.0
			for j := i - halfSmooth; j < i+halfSmooth; j++ {
				sum += hist[j]
			}
			histSmooth[i] = sum / float64(2*halfSmooth)
		}

		// Cap modal intensity at 95th percentile bin
		modalIntensity := 0
		for i, v := range histSmooth {
			if v > histSmooth[modalIntensity] {
				modalIntensity = i
			}
		}
		cap95 := int(0.95 * float64(histoSize))
		if modalIntensity > cap95 {
			modalIntensity = cap95
		}

		binEdge := func(bin int) float64 {
			return minVal + float64(bin)/float64(histoSize)*span
		}

		if N > 0 {
			totalCount := 0.0
			for _, v := range histSmooth {
				totalCount += v
			}
			countThresh := 0.68 * totalCount
			count := histSmooth[modalIntensity]
			indexBin := 1
			for count < countThresh {
				if modalIntensity+indexBin < histoSize {
					count += histSmooth[modalIntensity+indexBin]
				}
				if modalIntensity-indexBin >= 0 {
					count += histSmooth[modalIntensity-indexBin]
				}
				indexBin++
			}
			thresh := modalIntensity + int(N*float64(indexBin))
			if thresh >= histoSize {
				thresh = histoSize - 1
			}
			backgroundNoise[f] = binEdge(thresh)
		} else {
			backgroundNoise[f] = binEdge(modalIntensity)
		}
	}

	// Smooth background noise curve across frequency axis
	bgSmooth := make([]float64, numFreqs)
	for f := halfSmooth; f < numFreqs-halfSmooth; f++ {
		sum := 0.0
		for j := f - halfSmooth; j < f+halfSmooth; j++ {
			sum += backgroundNoise[j]
		}
		bgSmooth[f] = sum / float64(2*halfSmooth)
	}
	// Keep ends
	for f := 0; f < halfSmooth && f < numFreqs; f++ {
		bgSmooth[f] = backgroundNoise[f]
	}
	for f := numFreqs - halfSmooth; f < numFreqs; f++ {
		bgSmooth[f] = backgroundNoise[f]
	}

	// Subtract background and clip at lowValue
	newSpec := make([][]float64, numFreqs)
	for f := 0; f < numFreqs; f++ {
		newSpec[f] = make([]float64, numTimes)
		for t := 0; t < numTimes; t++ {
			v := working[f][t] - bgSmooth[f]
			if v < lowValue {
				v = lowValue
			}
			newSpec[f][t] = v
		}
	}
	return newSpec
}
