#pragma once
#ifndef AMSPORK3_SAMPLES_H
#define AMSPORK3_SAMPLES_H


#include <stdint.h>
#include <stdbool.h>

#ifndef exo_control_t
#define exo_control_t int_fast32_t
#endif
#if defined(__cplusplus)
static_assert(~exo_control_t(0) < 0, "exo_control_t must be signed");
#endif

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif



#ifdef __cplusplus
extern "C" {
#endif






#ifdef __cplusplus
}
#endif


#endif  // AMSPORK3_SAMPLES_H
