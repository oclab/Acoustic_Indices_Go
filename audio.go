package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/go-audio/wav"
)

// AudioFile holds decoded WAV audio data and metadata.
type AudioFile struct {
	FilePath string
	FileName string
	SR       int       // sample rate in Hz
	SigInt   []int     // PCM samples as integers
	SigFloat []float64 // normalized float samples in [-1, 1]
	Nyquist  float64
	Duration float64
}

// LoadAudioFile reads a WAV file and returns an AudioFile.
// Only the first channel is used for multi-channel files.
func LoadAudioFile(filePath string) (*AudioFile, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("cannot open %s: %w", filePath, err)
	}
	defer f.Close()

	decoder := wav.NewDecoder(f)
	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("cannot decode %s: %w", filePath, err)
	}
	if buf == nil || len(buf.Data) == 0 {
		return nil, fmt.Errorf("empty audio buffer in %s", filePath)
	}

	sr := int(decoder.SampleRate)
	bitDepth := int(decoder.BitDepth)
	numChannels := buf.Format.NumChannels
	if numChannels < 1 {
		numChannels = 1
	}

	numFrames := len(buf.Data) / numChannels
	sigInt := make([]int, numFrames)
	sigFloat := make([]float64, numFrames)

	absMax := float64(int(1) << (bitDepth - 1))

	for i := 0; i < numFrames; i++ {
		s := buf.Data[i*numChannels] // first channel only
		sigInt[i] = s
		sigFloat[i] = float64(s) / absMax
	}

	return &AudioFile{
		FilePath: filePath,
		FileName: filepath.Base(filePath),
		SR:       sr,
		SigInt:   sigInt,
		SigFloat: sigFloat,
		Nyquist:  float64(sr) / 2.0,
		Duration: float64(numFrames) / float64(sr),
	}, nil
}
