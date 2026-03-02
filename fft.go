package main

import (
	"math"
	"math/cmplx"
	"sync"
)

// fftInPlace computes an in-place Cooley-Tukey radix-2 DIT FFT.
// len(data) must be a power of 2.
func fftInPlace(data []complex128) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Bit-reversal permutation
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
	}

	// Butterfly stages
	for length := 2; length <= n; length <<= 1 {
		angle := -2.0 * math.Pi / float64(length)
		wn := complex(math.Cos(angle), math.Sin(angle))
		for i := 0; i < n; i += length {
			w := complex(1.0, 0.0)
			half := length / 2
			for k := 0; k < half; k++ {
				u := data[i+k]
				v := data[i+k+half] * w
				data[i+k] = u + v
				data[i+k+half] = u - v
				w *= wn
			}
		}
	}
}

// ifftInPlace computes an in-place inverse FFT.
func ifftInPlace(data []complex128) {
	n := len(data)
	// Conjugate inputs
	for i := range data {
		data[i] = cmplx.Conj(data[i])
	}
	fftInPlace(data)
	// Conjugate and scale outputs
	scale := complex(1.0/float64(n), 0)
	for i := range data {
		data[i] = cmplx.Conj(data[i]) * scale
	}
}

// bluesteinFFT computes the N-point DFT for arbitrary N using Bluestein's algorithm.
// Returns a slice of length N with the DFT values.
func bluesteinFFT(data []complex128) []complex128 {
	n := len(data)
	if n <= 1 {
		result := make([]complex128, n)
		copy(result, data)
		return result
	}
	// Chirp factors: w[k] = exp(-j*pi*k^2/N)
	w := make([]complex128, n)
	for k := 0; k < n; k++ {
		angle := math.Pi * float64(k) * float64(k) / float64(n)
		w[k] = complex(math.Cos(angle), -math.Sin(angle))
	}

	M := nextPow2(2*n - 1)

	// A[k] = data[k] * w[k]  (multiply by chirp)
	A := make([]complex128, M)
	for k := 0; k < n; k++ {
		A[k] = data[k] * w[k]
	}

	// B_padded: B[m] = conj(w[m]) = exp(+j*pi*m^2/N)
	// B[M-k] = B[-k] = B[k] (since B[m] is even in m^2)
	B := make([]complex128, M)
	for k := 0; k < n; k++ {
		B[k] = cmplx.Conj(w[k])
	}
	for k := 1; k < n; k++ {
		B[M-k] = cmplx.Conj(w[k])
	}
	fftInPlace(B) // pre-compute FFT of B

	fftInPlace(A)
	for k := range A {
		A[k] *= B[k]
	}
	ifftInPlace(A) // A now holds the convolution

	// X[k] = w[k] * conv[k]
	X := make([]complex128, n)
	for k := 0; k < n; k++ {
		X[k] = w[k] * A[k]
	}
	return X
}

// bluesteinPlan caches the precomputed chirp and FFT(B) arrays for a given (N, M) pair.
type bluesteinPlan struct {
	N int
	M int
	w []complex128 // chirp factors w[k] = exp(-j*pi*k^2/N)
	B []complex128 // FFT(B_padded), ready for pointwise multiplication
}

var (
	bluesteinCache = make(map[int]*bluesteinPlan)
	bluesteinMu    sync.RWMutex
)

// getBluesteinPlan returns (or creates) the precomputed plan for length N.
func getBluesteinPlan(n int) *bluesteinPlan {
	bluesteinMu.RLock()
	p, ok := bluesteinCache[n]
	bluesteinMu.RUnlock()
	if ok {
		return p
	}

	bluesteinMu.Lock()
	defer bluesteinMu.Unlock()
	// Double-check after acquiring write lock
	if p, ok = bluesteinCache[n]; ok {
		return p
	}

	M := nextPow2(2*n - 1)
	w := make([]complex128, n)
	for k := 0; k < n; k++ {
		angle := math.Pi * float64(k) * float64(k) / float64(n)
		w[k] = complex(math.Cos(angle), -math.Sin(angle))
	}
	B := make([]complex128, M)
	for k := 0; k < n; k++ {
		B[k] = cmplx.Conj(w[k])
	}
	for k := 1; k < n; k++ {
		B[M-k] = cmplx.Conj(w[k])
	}
	fftInPlace(B)

	p = &bluesteinPlan{N: n, M: M, w: w, B: B}
	bluesteinCache[n] = p
	return p
}

// bluesteinFFTFast uses a cached plan for the chirp/B arrays.
func bluesteinFFTFast(data []complex128) []complex128 {
	n := len(data)
	if n <= 1 {
		result := make([]complex128, n)
		copy(result, data)
		return result
	}
	p := getBluesteinPlan(n)
	A := make([]complex128, p.M)
	for k := 0; k < n; k++ {
		A[k] = data[k] * p.w[k]
	}
	fftInPlace(A)
	for k := range A {
		A[k] *= p.B[k]
	}
	ifftInPlace(A)
	X := make([]complex128, n)
	for k := 0; k < n; k++ {
		X[k] = p.w[k] * A[k]
	}
	return X
}

// fftGeneral computes the DFT for any length.
// Uses the fast in-place FFT for power-of-2 lengths, Bluestein otherwise.
func fftGeneral(data []complex128) []complex128 {
	n := len(data)
	if n&(n-1) == 0 { // power of 2
		result := make([]complex128, n)
		copy(result, data)
		fftInPlace(result)
		return result
	}
	return bluesteinFFTFast(data)
}


func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// hanningWindow returns a Hanning window of length n.
func hanningWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(n-1)))
	}
	return w
}

// hammingWindow returns a Hamming window of length n.
func hammingWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.54 - 0.46*math.Cos(2.0*math.Pi*float64(i)/float64(n-1))
	}
	return w
}

// blackmanWindow returns a Blackman window of length n.
func blackmanWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.42 - 0.5*math.Cos(2.0*math.Pi*float64(i)/float64(n-1)) +
			0.08*math.Cos(4.0*math.Pi*float64(i)/float64(n-1))
	}
	return w
}

// getWindow returns a window function by name.
func getWindow(name string, n int) []float64 {
	switch name {
	case "hamming":
		return hammingWindow(n)
	case "blackman":
		return blackmanWindow(n)
	default: // "hanning", "hann", or anything else
		return hanningWindow(n)
	}
}

// rfftMagnitude computes |FFT(frame)[0:halfLen]| for a real-valued frame.
// Returns halfLen = len(frame)/2 bins. Handles any frame length.
func rfftMagnitude(frame []float64, square bool) []float64 {
	n := len(frame)
	data := make([]complex128, n)
	for i, v := range frame {
		data[i] = complex(v, 0)
	}
	out := fftGeneral(data)
	half := n / 2
	mags := make([]float64, half)
	for k := 0; k < half; k++ {
		m := cmplx.Abs(out[k])
		if square {
			mags[k] = m * m
		} else {
			mags[k] = m
		}
	}
	return mags
}

// computeSpectrogram computes a power/amplitude spectrogram.
// Returns spectro[freqBin][timeFrame] and frequencies[freqBin].
func computeSpectrogram(
	sig []float64,
	sr int,
	windowLength, windowHop int,
	windowType string,
	square, normalized bool,
) ([][]float64, []float64) {
	W := getWindow(windowType, windowLength)
	half := windowLength / 2

	// Build frames
	var frames [][]float64
	for i := 0; i+windowLength <= len(sig); i += windowHop {
		frame := make([]float64, windowLength)
		for j := 0; j < windowLength; j++ {
			frame[j] = sig[i+j] * W[j]
		}
		frames = append(frames, frame)
	}
	if len(frames) == 0 {
		return nil, nil
	}

	numFrames := len(frames)
	// spectro[freq][time]
	spectro := make([][]float64, half)
	for f := range spectro {
		spectro[f] = make([]float64, numFrames)
	}

	for t, frame := range frames {
		mags := rfftMagnitude(frame, square)
		for f := 0; f < half; f++ {
			spectro[f][t] = mags[f]
		}
	}

	if normalized {
		maxVal := 0.0
		for _, row := range spectro {
			for _, v := range row {
				if v > maxVal {
					maxVal = v
				}
			}
		}
		if maxVal > 0 {
			for f := range spectro {
				for t := range spectro[f] {
					spectro[f][t] /= maxVal
				}
			}
		}
	}

	// Frequency axis: freq[i] = i * nyquist / (windowLength/2)
	nyquist := float64(sr) / 2.0
	frequencies := make([]float64, half)
	for i := range frequencies {
		frequencies[i] = float64(i) * nyquist / float64(half)
	}

	return spectro, frequencies
}

// hilbertEnvelope computes the envelope of sig via the analytic signal (Hilbert transform).
// Uses FFT padded to next power of 2.
func hilbertEnvelope(sig []float64) []float64 {
	n := nextPow2(len(sig))
	data := make([]complex128, n)
	for i, v := range sig {
		data[i] = complex(v, 0)
	}
	fftInPlace(data)

	// Apply Hilbert filter: double positive freqs, zero negative freqs
	// DC (0) and Nyquist (n/2) are unchanged
	for k := 1; k < n/2; k++ {
		data[k] *= 2
	}
	for k := n/2 + 1; k < n; k++ {
		data[k] = 0
	}

	ifftInPlace(data)

	// Return envelope (magnitude of analytic signal), trimmed to original length
	env := make([]float64, len(sig))
	for i := range env {
		env[i] = cmplx.Abs(data[i])
	}
	return env
}

// welchPSD estimates the power spectral density using Welch's method.
// Returns (frequencies, psd) with one-sided PSD using density scaling.
func welchPSD(sig []float64, sr, windowLength int) ([]float64, []float64) {
	overlap := windowLength / 2
	step := windowLength - overlap
	hamming := hammingWindow(windowLength)

	// Window normalization factor (sum of squares)
	winNorm := 0.0
	for _, v := range hamming {
		winNorm += v * v
	}

	nFreqs := windowLength/2 + 1
	psd := make([]float64, nFreqs)
	count := 0

	for start := 0; start+windowLength <= len(sig); start += step {
		// Detrend (subtract mean)
		mean := 0.0
		for j := 0; j < windowLength; j++ {
			mean += sig[start+j]
		}
		mean /= float64(windowLength)

		data := make([]complex128, windowLength)
		for j := 0; j < windowLength; j++ {
			data[j] = complex((sig[start+j]-mean)*hamming[j], 0)
		}
		out := fftGeneral(data)

		// Accumulate one-sided periodogram
		scale := float64(sr) * winNorm
		psd[0] += (real(out[0])*real(out[0]) + imag(out[0])*imag(out[0])) / scale
		for k := 1; k < nFreqs-1; k++ {
			psd[k] += 2.0 * (real(out[k])*real(out[k]) + imag(out[k])*imag(out[k])) / scale
		}
		nyqIdx := windowLength / 2
		psd[nyqIdx] += (real(out[nyqIdx])*real(out[nyqIdx]) + imag(out[nyqIdx])*imag(out[nyqIdx])) / scale

		count++
	}

	if count > 0 {
		for i := range psd {
			psd[i] /= float64(count)
		}
	}

	freqs := make([]float64, nFreqs)
	for i := range freqs {
		freqs[i] = float64(i) * float64(sr) / float64(windowLength)
	}
	return freqs, psd
}
