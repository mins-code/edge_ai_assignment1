// features.h -- the board's feature pipeline (Module 2 starter)
// It follows the definition in your handout exactly:
//   frame -> Hann (symmetric, numpy.hanning) -> zero-pad to nfft -> |FFT|^2 / nfft
//   -> HTK mel triangles, 20 Hz .. fs/2 -> 10*log10(max(m,1e-10)) -> optional DCT-II (ortho)
// Pure C++, no Arduino calls, so it also compiles on a laptop for testing.
//
// It is CORRECT but written the slow, textbook way (see dft() at the bottom).
// Mission 3: measure how long compute() takes on the board, decide whether that is
// fast enough, and if not, make it faster without changing a single output number
// by more than about 0.05 dB. Ask an LLM for hints, not code, and log what you asked.
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct FeatCfg {
  int fs = 16000, win = 480, hop = 320, nfft = 512, mels = 40, mfcc = 0;
};

class FeaturePipeline {
 public:
  FeatCfg c;
  bool ok = false;

  bool begin(const FeatCfg &cfg) {
    end();
    c = cfg;
    if (c.nfft < 16 || c.nfft > 2048 || (c.nfft & (c.nfft - 1)) || c.win > c.nfft || c.win < 2 ||
        c.hop < 1 || c.mels < 1 || c.mels > 128 || c.mfcc < 0 || c.mfcc > c.mels) return false;
    nb = c.nfft / 2 + 1;
    window = alloc(c.win); re = alloc(c.nfft); im = alloc(c.nfft); pw = alloc(nb);
    melw = alloc((size_t)c.mels * nb); melv = alloc(c.mels);
    tw_cos = alloc(c.nfft / 2); tw_sin = alloc(c.nfft / 2);
    bitrev = (int *)malloc(sizeof(int) * c.nfft);
   
    if (c.mfcc) dct = alloc((size_t)c.mfcc * c.mels);
    if (!window || !re || !im || !pw || !melw || !melv || !tw_cos || !tw_sin || !bitrev || (c.mfcc && !dct)) { end(); return false; }
    for (int k = 0; k < c.nfft / 2; k++) {
      float a = 2.0f * (float)M_PI * k / c.nfft;
      tw_cos[k] = cosf(a);
      tw_sin[k] = -sinf(a);
    }
    int log2n = 0;
    while ((1 << log2n) < c.nfft) log2n++;
    for (int i = 0; i < c.nfft; i++) {
      int r = 0;
      for (int b = 0; b < log2n; b++) if (i & (1 << b)) r |= 1 << (log2n - 1 - b);
      bitrev[i] = r;
    }
    for (int n = 0; n < c.win; n++) window[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / (c.win - 1));
    // HTK mel filterbank
    auto hz2mel = [](double f) { return 2595.0 * log10(1.0 + f / 700.0); };
    auto mel2hz = [](double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); };
    double m0 = hz2mel(20.0), m1 = hz2mel(c.fs / 2.0);
    double *edge = (double *)malloc(sizeof(double) * (c.mels + 2));
    for (int i = 0; i < c.mels + 2; i++) edge[i] = mel2hz(m0 + (m1 - m0) * i / (c.mels + 1));
    for (int m = 0; m < c.mels; m++)
      for (int k = 0; k < nb; k++) {
        double f = (double)k * c.fs / c.nfft, lo = edge[m], ce = edge[m + 1], hi = edge[m + 2], w = 0;
        if (f > lo && f <= ce) w = (f - lo) / (ce - lo);
        else if (f > ce && f < hi) w = (hi - f) / (hi - ce);
        melw[(size_t)m * nb + k] = (float)w;
      }
    free(edge);
    for (int q = 0; q < c.mfcc; q++)
      for (int m = 0; m < c.mels; m++)
        dct[(size_t)q * c.mels + m] = (float)((q == 0 ? sqrt(1.0 / c.mels) : sqrt(2.0 / c.mels)) *
                                              cos(M_PI * q * (2 * m + 1) / (2.0 * c.mels)));
    ok = true;
    return true;
  }

  int outSize() const { return c.mfcc ? c.mfcc : c.mels; }

  // frame: c.win float samples (already dequantised). out: outSize() floats.
  void compute(const float *frame, float *out) {
    for (int n = 0; n < c.win; n++) { re[n] = frame[n] * window[n]; im[n] = 0; }
    for (int n = c.win; n < c.nfft; n++) { re[n] = 0; im[n] = 0; }
    dft();
    for (int k = 0; k < nb; k++) pw[k] = (re[k] * re[k] + im[k] * im[k]) / c.nfft;
    for (int m = 0; m < c.mels; m++) {
      const float *w = melw + (size_t)m * nb; float s = 0;
      for (int k = 0; k < nb; k++) s += w[k] * pw[k];
      melv[m] = 10.0f * log10f(s > 1e-10f ? s : 1e-10f);
    }
    if (!c.mfcc) { memcpy(out, melv, sizeof(float) * c.mels); return; }
    for (int q = 0; q < c.mfcc; q++) {
      const float *d = dct + (size_t)q * c.mels; float s = 0;
      for (int m = 0; m < c.mels; m++) s += d[m] * melv[m];
      out[q] = s;
    }
  }

  void end() {
    float **ps[] = {&window, &re, &im, &pw, &melw, &melv, &dct, &tw_cos, &tw_sin};
    for (auto p : ps) { free(*p); *p = nullptr; }
    free(bitrev); bitrev = nullptr;
    ok = false;
  }
  ~FeaturePipeline() { end(); }

 private:
  int nb = 0;
  int *bitrev = nullptr;
  float *window = nullptr, *re = nullptr, *im = nullptr, *pw = nullptr, *melw = nullptr,
        *melv = nullptr, *dct = nullptr, *tw_cos = nullptr, *tw_sin = nullptr;

  static float *alloc(size_t n) { return (float *)malloc(sizeof(float) * n); }

  // In-place radix-2 Cooley-Tukey FFT using precomputed twiddle factors.
  void dft() {
    for (int i = 0; i < c.nfft; i++) {
      int j = bitrev[i];
      if (i < j) {
        float tr = re[i]; re[i] = re[j]; re[j] = tr;
      }
    }
    for (int len = 2; len <= c.nfft; len <<= 1) {
      int half = len >> 1;
      int step = c.nfft / len;
      for (int i = 0; i < c.nfft; i += len) {
        for (int j = 0; j < half; j++) {
          float wr = tw_cos[j * step], wi = tw_sin[j * step];
          int u = i + j, v = u + half;
          float tr = wr * re[v] - wi * im[v];
          float ti = wr * im[v] + wi * re[v];
          re[v] = re[u] - tr; im[v] = im[u] - ti;
          re[u] += tr; im[u] += ti;
        }
      }
    }
  }
};
