#include <stdio.h>
#include <inttypes.h>
#include <math.h>

typedef uint16_t fpt;
#define FIXED_POINT_FRACTIONAL_BITS 5
float fixed_to_float(fpt input) {
    return ((float)input / (float)(1 << FIXED_POINT_FRACTIONAL_BITS));
}
fpt float_to_fixed(float input) {
    return (fpt)(round(input * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

void main(void){
  float x = 0.003;

  fpt fpt1 = float_to_fixed(x); 
  printf("fpt1 = %d float fpt1 = %f\n", fpt1, fixed_to_float(fpt1));
}
