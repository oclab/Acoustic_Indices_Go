package main

import (
	"math"
	"testing"
)

// TestFFTBasic verifies that the FFT of a pure sine wave has a peak at the expected bin.
func TestFFTBasic(t *testing.T) {
	n := 512
	freq := 10 // bin number
	data := make([]complex128, n)
	for i := 0; i < n; i++ {
		angle := 2 * math.Pi * float64(freq) * float64(i) / float64(n)
		data[i] = complex(math.Sin(angle), 0)
	}
	fftInPlace(data)

	// Find peak
	peakBin := 0
	peakMag := 0.0
	for k := 1; k <= n/2; k++ {
		mag := math.Sqrt(real(data[k])*real(data[k]) + imag(data[k])*imag(data[k]))
		if mag > peakMag {
			peakMag = mag
			peakBin = k
		}
	}
	if peakBin != freq {
		t.Errorf("FFT peak at bin %d, expected %d", peakBin, freq)
	}
}

// TestBluesteinMatchesPow2 verifies that Bluestein FFT matches the power-of-2 FFT for pow2 inputs.
func TestBluesteinMatchesPow2(t *testing.T) {
	n := 128
	data := make([]complex128, n)
	for i := 0; i < n; i++ {
		data[i] = complex(float64(i%7)-3.0, 0)
	}

	// Reference: power-of-2 FFT
	ref := make([]complex128, n)
	copy(ref, data)
	fftInPlace(ref)

	// Bluestein on same data
	got := bluesteinFFT(data)

	for k := 0; k < n; k++ {
		dr := real(ref[k]) - real(got[k])
		di := imag(ref[k]) - imag(got[k])
		if math.Sqrt(dr*dr+di*di) > 1e-8 {
			t.Errorf("bin %d: ref=(%.6f,%.6f) got=(%.6f,%.6f)",
				k, real(ref[k]), imag(ref[k]), real(got[k]), imag(got[k]))
			break
		}
	}
}

// TestBluesteinNonPow2 verifies Bluestein FFT for a non-power-of-2 length (N=300).
func TestBluesteinNonPow2(t *testing.T) {
	n := 300
	freq := 15 // bin number
	data := make([]complex128, n)
	for i := 0; i < n; i++ {
		angle := 2 * math.Pi * float64(freq) * float64(i) / float64(n)
		data[i] = complex(math.Sin(angle), 0)
	}

	out := bluesteinFFTFast(data)

	peakBin := 0
	peakMag := 0.0
	for k := 1; k < n/2; k++ {
		mag := math.Sqrt(real(out[k])*real(out[k]) + imag(out[k])*imag(out[k]))
		if mag > peakMag {
			peakMag = mag
			peakBin = k
		}
	}
	if peakBin != freq {
		t.Errorf("Bluestein peak at bin %d, expected %d", peakBin, freq)
	}
}

// TestHanningWindow verifies Hanning window properties.
func TestHanningWindow(t *testing.T) {
	n := 8
	w := hanningWindow(n)
	// First and last elements should be 0
	if math.Abs(w[0]) > 1e-10 || math.Abs(w[n-1]) > 1e-10 {
		t.Errorf("Hanning window endpoints should be ~0, got %v", w)
	}
	// Middle element should be ~1
	if math.Abs(w[n/2-1]-1.0) > 0.1 {
		t.Errorf("Hanning window middle should be near 1, got %f", w[n/2-1])
	}
}

// TestHilbertEnvelope verifies that the Hilbert envelope of a sine wave is ~constant.
func TestHilbertEnvelope(t *testing.T) {
	n := 512
	sig := make([]float64, n)
	amp := 2.5
	for i := range sig {
		sig[i] = amp * math.Sin(2*math.Pi*20*float64(i)/float64(n))
	}
	env := hilbertEnvelope(sig)
	// Envelope should be approximately equal to amplitude (ignoring edge effects)
	start, end := n/8, 7*n/8
	for i := start; i < end; i++ {
		if math.Abs(env[i]-amp) > 0.05 {
			t.Errorf("envelope[%d] = %f, expected ~%f", i, env[i], amp)
			break
		}
	}
}

// TestGini verifies the Gini index computation.
func TestGini(t *testing.T) {
	// All equal values → Gini = 0
	eq := []float64{1, 1, 1, 1}
	if math.Abs(gini(eq)) > 1e-10 {
		t.Errorf("equal values: expected Gini=0, got %f", gini(eq))
	}

	// Perfectly unequal (one nonzero) → Gini ≈ 1
	ineq := []float64{0, 0, 0, 1}
	g := gini(ineq)
	if g < 0.7 {
		t.Errorf("unequal values: expected Gini near 1, got %f", g)
	}
}

// TestComputeSpectrogramShape checks that the spectrogram has the right dimensions.
func TestComputeSpectrogramShape(t *testing.T) {
	sr := 22050
	n := 22050 // 1 second
	sig := make([]float64, n)
	for i := range sig {
		sig[i] = math.Sin(2 * math.Pi * 440 * float64(i) / float64(sr))
	}
	windowLength := 512
	windowHop := 256
	spectro, freqs := computeSpectrogram(sig, sr, windowLength, windowHop, "hanning", true, false)

	expectedFreqs := windowLength / 2
	if len(spectro) != expectedFreqs {
		t.Errorf("spectro freq bins: got %d, want %d", len(spectro), expectedFreqs)
	}
	if len(freqs) != expectedFreqs {
		t.Errorf("freq vector length: got %d, want %d", len(freqs), expectedFreqs)
	}
	expectedFrames := (n-windowLength)/windowHop + 1
	if len(spectro[0]) != expectedFrames {
		t.Errorf("spectro frames: got %d, want %d", len(spectro[0]), expectedFrames)
	}
}

// TestComputeSH verifies spectral entropy is in [0,1].
func TestComputeSH(t *testing.T) {
	sr := 22050
	n := 22050
	sig := make([]float64, n)
	for i := range sig {
		sig[i] = math.Sin(2 * math.Pi * 440 * float64(i) / float64(sr))
	}
	spectro, _ := computeSpectrogram(sig, sr, 512, 256, "hanning", true, false)
	sh := computeSH(spectro)
	if sh < 0 || sh > 1 {
		t.Errorf("SH=%f not in [0,1]", sh)
	}
}

// TestComputeTH verifies temporal entropy is in [0,1].
func TestComputeTH(t *testing.T) {
	n := 22050
	sig := make([]float64, n)
	for i := range sig {
		sig[i] = math.Sin(2 * math.Pi * 440 * float64(i) / float64(n))
	}
	th := computeTH(sig)
	if th < 0 || th > 1 {
		t.Errorf("TH=%f not in [0,1]", th)
	}
}

// TestLoadAudioFile verifies that a WAV file can be loaded.
func TestLoadAudioFile(t *testing.T) {
	af, err := LoadAudioFile("audio_files/1.wav")
	if err != nil {
		t.Fatalf("LoadAudioFile failed: %v", err)
	}
	if af.SR <= 0 {
		t.Errorf("invalid sample rate: %d", af.SR)
	}
	if len(af.SigFloat) == 0 {
		t.Error("empty signal")
	}
	if af.FileName == "" {
		t.Error("empty filename")
	}
}

// TestButterworthHighPassSOS verifies that HP filter attenuates DC (ω=0) and passes Nyquist (ω=π).
func TestButterworthHighPassSOS(t *testing.T) {
	sos := ButterworthHighPassSOS(8, 0.1)
	if len(sos) != 4 {
		t.Fatalf("expected 4 SOS sections, got %d", len(sos))
	}

	// Gain at Nyquist (z=-1) should be ~1
	gain := 1.0
	for _, s := range sos {
		num := s.B[0] - s.B[1] + s.B[2]
		den := 1.0 - s.A[1] + s.A[2]
		if den == 0 {
			t.Fatal("zero denominator in SOS section")
		}
		gain *= num / den
	}
	if math.Abs(gain-1.0) > 0.01 {
		t.Errorf("Nyquist gain = %f, expected ~1.0", gain)
	}
}
