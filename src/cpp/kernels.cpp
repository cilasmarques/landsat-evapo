#include "kernels.h"

void rah_correction_cycle_STEEP(int start_line, int end_line, int width_band, float a, float b, float *surface_temperature_pointer,
                                float *d0_pointer, float *zom_pointer, float *kb1_pointer, float *sensible_heat_flux_pointer, 
                                float *ustar_pointer, float *aerodynamic_resistance_pointer)
{
  for (int line = start_line; line < end_line; line++)
  {
    for (int col = 0; col < width_band; col++)
    {
      int pos = line * width_band + col;

      float DISP = d0_pointer[pos];
      float dT_ini_terra = (a + b * (surface_temperature_pointer[pos] - 273.15));

      // H_ini_terra
      sensible_heat_flux_pointer[pos] = RHO * SPECIFIC_HEAT_AIR * (dT_ini_terra) / aerodynamic_resistance_pointer[pos];

      // L_MB_terra
      float ustar_pow_3 = ustar_pointer[pos] * ustar_pointer[pos] * ustar_pointer[pos];

      float L = -1 * ((RHO * SPECIFIC_HEAT_AIR * ustar_pow_3 * surface_temperature_pointer[pos]) / (VON_KARMAN * GRAVITY * sensible_heat_flux_pointer[pos]));

      float y2 = pow((1 - (16 * (10 - DISP)) / L), 0.25);
      float x200 = pow((1 - (16 * (10 - DISP)) / L), 0.25);

      float psi2, psi200;
      if (!isnan(L) && L > 0)
      {
        psi2 = -5 * ((10 - DISP) / L);
        psi200 = -5 * ((10 - DISP) / L);
      }
      else
      {
        psi2 = 2 * log((1 + y2 * y2) / 2);
        psi200 = 2 * log((1 + x200) / 2) + log((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;
      }

      // u*
      float ust = (VON_KARMAN * ustar_pointer[pos]) / (log((10 - DISP) / zom_pointer[pos]) - psi200);

      // rah
      float zoh_terra = zom_pointer[pos] / pow(exp(1.0), (kb1_pointer[pos]));
      float temp_rah1_corr_terra = (ust * VON_KARMAN);
      float temp_rah2_corr_terra = log((10 - DISP) / zom_pointer[pos]) - psi2;
      float temp_rah3_corr_terra = temp_rah1_corr_terra * log(zom_pointer[pos] / zoh_terra);
      float rah = (temp_rah1_corr_terra * temp_rah2_corr_terra) + temp_rah3_corr_terra;

      ustar_pointer[pos] = ust;
      aerodynamic_resistance_pointer[pos] = rah;
    }
  }
};
