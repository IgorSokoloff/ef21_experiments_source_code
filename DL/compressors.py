#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random, math

class CompressorType:
  IDENTICAL                = 1 # Identical compressor
  LAZY_COMPRESSOR          = 2 # Lazy or Bernulli compressor
  RANDK_COMPRESSOR         = 3 # Rank-K compressor
  NATURAL_COMPRESSOR_FP64  = 4 # Natural compressor with FP64
  NATURAL_COMPRESSOR_FP32  = 5 # Natural compressor with FP32
  STANDARD_DITHERING_FP64  = 6 # Standard dithering with FP64
  STANDARD_DITHERING_FP32  = 7 # Standard dithering with FP32
  NATURAL_DITHERING_FP32   = 8 # Natural Dithering applied for FP32 components vectors
  NATURAL_DITHERING_FP64   = 9 # Natural Dithering applied for FP64 components vectors
  TOPK_COMPRESSOR         = 10 # Top-K compressor

class Compressor:
    def __init__(self, compressorName = ""):
        self.compressorName = compressorName
        self.compressorType = CompressorType.IDENTICAL
        self.w = 0.0
        self.alpha = 0.0
        self.total_input_components = 0
        self.really_need_to_send_components = 0
        self.last_input_advance = 0
        self.last_need_to_send_advance = 0

    def name(self):
        omega = r'$\omega$'
        if self.compressorType == CompressorType.IDENTICAL: return f"Identical"
        if self.compressorType == CompressorType.LAZY_COMPRESSOR: return f"Bernoulli(Lazy) [p={self.P:g},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.RANDK_COMPRESSOR: return f" (K={self.K})"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP64: return f"Natural for fp64 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32: return f"Natural for fp32 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp64[s={self.s}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp32[s={self.s}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP32:  return f"Natural Dithering for fp32[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP64:  return f"Natural Dithering for fp64[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.TOPK_COMPRESSOR: return f" Top (K={self.K})"
        return "?"

    def fullName(self):
        omega = r'$\omega$'
        if self.compressorType == CompressorType.IDENTICAL: return f"Identical"
        if self.compressorType == CompressorType.LAZY_COMPRESSOR: return f"Bernoulli(Lazy) [p={self.P:g},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.RANDK_COMPRESSOR: return f"Rand [K={self.K},D={self.D}]"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP64: return f"Natural for fp64 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32: return f"Natural for fp32 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp64[s={self.s}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp32[s={self.s}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP32:  return f"Natural Dithering for fp32[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP64:  return f"Natural Dithering for fp64[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.TOPK_COMPRESSOR: return f"Top [K={self.K},D={self.D}]"
        return "?"

    def resetStats(self):
        self.total_input_components = 0
        self.really_need_to_send_components = 0
        self.last_input_advance = 0
        self.last_need_to_send_advance = 0

    def makeIdenticalCompressor(self):
        self.compressorType = CompressorType.IDENTICAL
        self.w = 0.0
        self.resetStats()

    def makeLazyCompressor(self, P):
        # w + 1 = p* 1/(p**2) => w = 1/p - 1
        self.compressorType = CompressorType.LAZY_COMPRESSOR
        self.P = P
        self.w = 1.0 / P - 1.0
        self.resetStats()


    def makeStandardDitheringFP64(self, levels, vectorNormCompressor, p = float("inf")):
        self.compressorType = CompressorType.STANDARD_DITHERING_FP64
        self.levelsValues = np.arange(0.0, 1.1, 1.0/levels)     # levels + 1 values in range [0.0, 1.0] which uniformly split this segment
        self.s = len(self.levelsValues) - 1                     # # should be equal to level
        assert self.s == levels

        self.p = p
        self.vectorNormCompressor = vectorNormCompressor
        self.w = 0.0 # TODO

        self.resetStats()

    def makeStandardDitheringFP32(self, levels, vectorNormCompressor, p = float("inf")):
        self.compressorType = CompressorType.STANDARD_DITHERING_FP32
        self.levelsValues = torch.arange(0.0, 1.1, 1.0/levels)     # levels + 1 values in range [0.0, 1.0] which uniformly split this segment
        self.s = len(self.levelsValues) - 1                        # should be equal to level
        assert self.s == levels

        self.p = p
        self.vectorNormCompressor = vectorNormCompressor
        self.w = 0.0 # TODO

        self.resetStats()

    def makeQSGD_FP64(self, levels, dInput):
        norm_compressor = Compressor("norm_compressor")
        norm_compressor.makeIdenticalCompressor()
        self.makeStandardDitheringFP64(levels, norm_compressor, p = 2)
        # Lemma 3.1. from https://arxiv.org/pdf/1610.02132.pdf, page 5
        self.w = min(dInput/(levels*levels), dInput**0.5/levels)

    def makeNaturalDitheringFP64(self, levels, dInput, p = float("inf")):
        self.compressorType = CompressorType.NATURAL_DITHERING_FP64
        self.levelsValues = torch.zeros(levels + 1)
        for i in range(levels):
            self.levelsValues[i] = (1.0/2.0)**i
        self.levelsValues = torch.flip(self.levelsValues, dims = [0])
        self.s = len(self.levelsValues) - 1
        assert self.s == levels

        self.p = p

        r = min(p, 2)
        self.w = 1.0/8.0 + (dInput** (1.0/r)) / (2**(self.s - 1)) * min(1, (dInput**(1.0/r)) / (2**(self.s-1)))
        self.resetStats()

    def makeNaturalDitheringFP32(self, levels, dInput, p = float("inf")):
        self.compressorType = CompressorType.NATURAL_DITHERING_FP32
        self.levelsValues = torch.zeros(levels + 1)
        for i in range(levels):
            self.levelsValues[i] = (1.0/2.0)**i
        self.levelsValues = torch.flip(self.levelsValues, dims=[0])
        self.s = len(self.levelsValues) - 1
        assert self.s == levels

        self.p = p

        r = min(p, 2)
        self.w = 1.0/8.0 + (dInput** (1.0/r)) / (2**(self.s - 1)) * min(1, (dInput**(1.0/r)) / (2**(self.s-1)))
        self.resetStats()

    # K - how much component we leave from input vector
    # D - input vector dimension
    def makeRandKCompressor(self, K, D):
        # E[|C(x)|^2]=(d*d)/(k*k) * E[sum( (I |xi|)^2)] = (d*d)/(k*k) * k/d *|x|^2 = d/k * (x^2) = (w + 1) (x^2) => w = d/k-1
        self.compressorType = CompressorType.RANDK_COMPRESSOR
        self.D = D
        self.K = K
        self.w = self.D / self.K - 1.0
        self.resetStats()

    def makeTopKCompressor(self, K, D):
        self.compressorType = CompressorType.TOPK_COMPRESSOR
        self.D = D
        self.K = K
        self.alpha = self.K / self.D
        self.resetStats()

    def makeNaturalCompressorFP64(self):
        self.compressorType = CompressorType.NATURAL_COMPRESSOR_FP64
        self.w = 1.0/8.0
        self.resetStats()

    def makeNaturalCompressorFP32(self):
        self.compressorType = CompressorType.NATURAL_COMPRESSOR_FP32
        self.w = 1.0/8.0
        self.resetStats()

    def getW(self):
        return self.w

    def getAlpha(self):
        return self.alpha

    def compressVector(self, x):
        d = max(x.shape)

        self.last_input_advance = d
        self.last_need_to_send_advance = 0

        if self.compressorType == CompressorType.IDENTICAL:
            out = x
            self.last_need_to_send_advance = d
        elif self.compressorType == CompressorType.LAZY_COMPRESSOR:
            testp = random.random()
            if testp < self.P:
                out = x / (self.P)
                self.last_need_to_send_advance = d
            else:
                out = torch.zeros_like(x)
                self.last_need_to_send_advance = 0
        elif self.compressorType == CompressorType.RANDK_COMPRESSOR:
            #S = torch.arange(self.D)
            # np.random.shuffle(S)
            S = torch.randperm(self.D)
            S = S[0:self.K]
            out = torch.zeros_like(x)
            out[S] = self.D / self.K * x[S]
            self.last_need_to_send_advance = self.K

        elif self.compressorType == CompressorType.TOPK_COMPRESSOR:
            #S = torch.arange(self.D)
            # np.random.shuffle(S)
            _, S = torch.topk(torch.abs(x), self.K)
            out = torch.zeros_like(x)
            out[S] = x[S]
            self.last_need_to_send_advance = self.K

        elif self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP64 or self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32:
            out = torch.zeros_like(x)
            for i in range(0, d):
                if x[i] == 0.0:
                    out[i] = 0.0
                else:
                    sign = torch.sign(x[i])
                    alpha = torch.log2(torch.abs(x[i]))
                    alpha_down = torch.floor(alpha)
                    alpha_up = torch.ceil(alpha)
                    pt = (2**(alpha_up) - torch.abs(x[i])) / (2**alpha_down)
                    testp = random.random()
                    if testp < pt:
                        out[i] = sign * (2**alpha_down)
                    else:
                        out[i] = sign * (2**alpha_up)

            if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP64:
                self.last_need_to_send_advance = 12.0/64.0 * d                  # 11 bits for the exponent and 1 bit for the sign
            elif self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32:
                self.last_need_to_send_advance = 9.0/32.0 * d                   # 8-bit in exponent and extra bit of sign

        elif self.compressorType == CompressorType.STANDARD_DITHERING_FP64 or self.compressorType == CompressorType.STANDARD_DITHERING_FP32:
            out = torch.zeros_like(x)
            pnorm = torch.norm(x, p=self.p)

            pnorm_to_send = self.vectorNormCompressor.compressVector(torch.Tensor([pnorm]))[0]
            self.last_need_to_send_advance = 0
            # Sending pnorm
            self.last_need_to_send_advance = self.vectorNormCompressor.last_need_to_send_advance

            for i in range(0, d):
                if x[i] == 0.0:
                    out[i] = 0.0
                else:
                    sign = torch.sign(x[i])
                    yi = torch.abs(x[i])/pnorm

                    for s in range(len(self.levelsValues)):
                        if yi >= self.levelsValues[s] and yi <= self.levelsValues[s+1]:
                            p = (yi - self.levelsValues[s+1])/(self.levelsValues[s] - self.levelsValues[s+1])
                            testp = random.random()
                            if testp < p:
                                out[i] = self.levelsValues[s]
                            else:
                                out[i] = self.levelsValues[s + 1]
                            break

                    # To emulate that out is reconstitute
                    out[i] =  out[i] * sign * pnorm

                    # Calculate need send items
                    if self.compressorType == CompressorType.STANDARD_DITHERING_FP64:
                        # items to send compressed norm + 1 bit for sign log2(levels) bits to send level
                        self.last_need_to_send_advance += (1.0 + math.ceil(math.log2(self.s)))/64.0
                    elif self.compressorType == CompressorType.STANDARD_DITHERING_FP32:
                        # items to send compressed norm + 1 bit for sign log2(levels) bits to send level
                        self.last_need_to_send_advance += (1.0 + math.ceil(math.log2(self.s)))/32.0

        elif self.compressorType == CompressorType.NATURAL_DITHERING_FP64 or self.compressorType == CompressorType.NATURAL_DITHERING_FP32:
            out = torch.zeros_like(x)
            pnorm = torch.norm(x, p=self.p)
            pnorm_to_send = pnorm
            self.last_need_to_send_advance = 1

            for i in range(0, d):
                if x[i] == 0.0:
                    out[i] = 0.0
                else:
                    sign = torch.sign(x[i])
                    yi = torch.abs(x[i])/pnorm

                    for s in range(len(self.levelsValues)):
                        if yi >= self.levelsValues[s] and yi <= self.levelsValues[s+1]:
                            p = (yi - self.levelsValues[s+1])/(self.levelsValues[s] - self.levelsValues[s+1])
                            testp = random.random()
                            if testp < p:
                                out[i] = self.levelsValues[s]
                            else:
                                out[i] = self.levelsValues[s + 1]
                            break

                    # To emulate that out is reconstitute
                    out[i] =  out[i] * sign * pnorm

            # Calculate need send items
            if self.compressorType == CompressorType.NATURAL_DITHERING_FP64:
                self.last_need_to_send_advance = d*(1.0 + math.ceil(math.log2(self.s)))/64.0               # 1 bit for sign log2(levels) bits to send level
            elif self.compressorType == CompressorType.NATURAL_DITHERING_FP32:
                self.last_need_to_send_advance = d*(1.0 + math.ceil(math.log2(self.s)))/32.0               # 1 bit for sign log2(levels) bits to send level

        # update stats about sending components
        self.really_need_to_send_components += self.last_need_to_send_advance
        self.total_input_components += self.last_input_advance

        return out
