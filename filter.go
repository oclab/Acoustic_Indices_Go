package main

import (
	"math"
	"math/cmplx"
)

// SOSSection represents one second-order IIR section.
// Transfer function: H(z) = (b[0] + b[1]*z^{-1} + b[2]*z^{-2}) /
//
//	(1   + a[1]*z^{-1} + a[2]*z^{-2})
type SOSSection struct {
	B [3]float64
	A [3]float64 // A[0] is implicitly 1
}

// ButterworthHighPassSOS designs a Butterworth high-pass filter.
// order must be even; Wn is the normalized cutoff in (0, 1) where 1 = Nyquist.
// Returns second-order sections with unity gain at Nyquist (ω=π).
func ButterworthHighPassSOS(order int, Wn float64) []SOSSection {
	if order%2 != 0 {
		// Pad to even order by incrementing (conservative)
		order++
	}
	numSections := order / 2

	// Prewarped analog cutoff frequency (bilinear transform prewarping)
	Wd := 2.0 * math.Tan(math.Pi*Wn/2.0)

	sections := make([]SOSSection, numSections)
	for k := 0; k < numSections; k++ {
		// Butterworth LP prototype pole (conjugate pair representative):
		// p_k = exp(j*pi*(2k+order+1)/(2*order)) for k=0..numSections-1
		angle := math.Pi * float64(2*k+order+1) / float64(2*order)
		pLP := complex(math.Cos(angle), math.Sin(angle))

		// LP → HP analog transformation: p_hp = Wd / p_lp
		pHP := complex(Wd, 0) / pLP
		pHPconj := cmplx.Conj(pHP)

		// Bilinear transform to digital poles: z = (1 + p/2)/(1 - p/2)
		z1 := (1 + pHP/2) / (1 - pHP/2)
		z2 := (1 + pHPconj/2) / (1 - pHPconj/2)

		// Verify z2 ≈ conj(z1) (numerical check is implicit)
		_ = z2

		// Denominator coefficients from (z - z1)(z - z2)
		// = z^2 - 2*Re(z1)*z + |z1|^2
		reZ1 := real(z1)
		magZ1Sq := real(z1)*real(z1) + imag(z1)*imag(z1)

		// Numerator: HP digital zeros are at z=1 (from analog zeros at s=0)
		// (z - 1)^2 = z^2 - 2z + 1  →  [1, -2, 1] in z powers
		//  in z^{-1} form: 1 - 2*z^{-1} + z^{-2}

		// Gain of this section at Nyquist (z = -1, ω = π):
		//   Num(-1) = 1 + 2 + 1 = 4
		//   Den(-1) = 1 + 2*Re(z1) + |z1|^2
		denAtNyquist := 1.0 + 2.0*reZ1 + magZ1Sq
		gain := 4.0 / denAtNyquist

		// Normalize numerator for unity gain at Nyquist
		sections[k] = SOSSection{
			B: [3]float64{1.0 / gain, -2.0 / gain, 1.0 / gain},
			A: [3]float64{1.0, -2.0 * reZ1, magZ1Sq},
		}
	}
	return sections
}

// sosfiltForward applies SOS filter to sig in the forward direction.
func sosfiltForward(sos []SOSSection, sig []float64) []float64 {
	out := make([]float64, len(sig))
	copy(out, sig)

	for _, s := range sos {
		// Direct Form II transposed
		w0, w1 := 0.0, 0.0
		tmp := make([]float64, len(out))
		for n, x := range out {
			y := s.B[0]*x + w0
			w0 = s.B[1]*x - s.A[1]*y + w1
			w1 = s.B[2]*x - s.A[2]*y
			tmp[n] = y
		}
		out = tmp
	}
	return out
}

// filtfilt applies zero-phase IIR filtering (forward + backward pass).
// Uses edge-value padding to reduce transients.
func filtfilt(sos []SOSSection, sig []float64) []float64 {
	// Pad length: 3 * max filter order
	padLen := 3 * 2 * len(sos) // 3 * (2 * numSections) samples of padding
	if padLen >= len(sig) {
		padLen = len(sig) - 1
	}

	// Reflect padding at both ends
	padded := make([]float64, padLen+len(sig)+padLen)
	// Left edge: reflect sig[0:padLen] reversed around sig[0]
	for i := 0; i < padLen; i++ {
		padded[i] = 2*sig[0] - sig[padLen-i]
	}
	copy(padded[padLen:], sig)
	// Right edge: reflect sig[len-padLen:len] reversed around sig[len-1]
	for i := 0; i < padLen; i++ {
		padded[padLen+len(sig)+i] = 2*sig[len(sig)-1] - sig[len(sig)-2-i]
	}

	// Forward pass
	fwd := sosfiltForward(sos, padded)

	// Reverse
	rev := make([]float64, len(fwd))
	for i, v := range fwd {
		rev[len(fwd)-1-i] = v
	}

	// Backward pass
	bwd := sosfiltForward(sos, rev)

	// Reverse again and trim padding
	result := make([]float64, len(sig))
	for i := range result {
		result[i] = bwd[len(bwd)-1-padLen-i]
	}
	return result
}

// ApplyButterworthHP applies a Butterworth high-pass filter to the signal.
// order: filter order; freqHz: cutoff frequency in Hz; sr: sample rate in Hz.
func ApplyButterworthHP(sig []float64, sr int, order int, freqHz float64) []float64 {
	nyquist := float64(sr) / 2.0
	Wn := freqHz / nyquist
	if Wn <= 0 || Wn >= 1 {
		return sig
	}
	sos := ButterworthHighPassSOS(order, Wn)
	return filtfilt(sos, sig)
}
