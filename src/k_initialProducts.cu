#include "kernels.cuh"

__device__ int width_d;
__device__ int height_d;

__global__ void rad_kernel(half *band_d, half *radiance_d, half *rad_add_d, half *rad_mult_d, int band_idx)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        radiance_d[pos] = band_d[pos] * rad_mult_d[band_idx] + rad_add_d[band_idx];

        if (__half2float(radiance_d[pos]) <= 0)
            radiance_d[pos] = NAN;
    }
}

__global__ void ref_kernel(half *band_d, half *reflectance_d, half *ref_add_d, half *ref_mult_d, float sin_sun, int band_idx)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        reflectance_d[pos] = __float2half((__half2float(band_d[pos]) * __half2float(ref_mult_d[band_idx]) + __half2float(ref_add_d[band_idx])) / sin_sun);

        if (__half2float(reflectance_d[pos]) <= 0)
            reflectance_d[pos] = NAN;
    }
}

__global__ void albedo_kernel(half *reflectance_blue_d, half *reflectance_green_d, half *reflectance_red_d, half *reflectance_nir_d, half *reflectance_swir1_d, half *reflectance_swir2_d, half *tal_d, half *albedo_d, half *ref_w_coeff_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        float alb_toa = reflectance_blue_d[pos] * ref_w_coeff_d[PARAM_BAND_BLUE_INDEX] +
                        reflectance_green_d[pos] * ref_w_coeff_d[PARAM_BAND_GREEN_INDEX] +
                        reflectance_red_d[pos] * ref_w_coeff_d[PARAM_BAND_RED_INDEX] +
                        reflectance_nir_d[pos] * ref_w_coeff_d[PARAM_BAND_NIR_INDEX] +
                        reflectance_swir1_d[pos] * ref_w_coeff_d[PARAM_BAND_SWIR1_INDEX] +
                        reflectance_swir2_d[pos] * ref_w_coeff_d[PARAM_BAND_SWIR2_INDEX];

        albedo_d[pos] = __float2half((alb_toa - 0.03) / (__half2float(tal_d[pos]) * __half2float(tal_d[pos])));

        if (__half2float(albedo_d[pos]) <= 0)
            albedo_d[pos] = NAN;
    }
}

__global__ void ndvi_kernel(half *reflectance_nir_d, half *reflectance_red_d, half *ndvi_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        ndvi_d[pos] = (reflectance_nir_d[pos] - reflectance_red_d[pos]) / (reflectance_nir_d[pos] + reflectance_red_d[pos]);

        if (__half2float(ndvi_d[pos]) <= -1.0f || __half2float(ndvi_d[pos]) >= 1.0f)
            ndvi_d[pos] = NAN;
    }
}

__global__ void pai_kernel(half *reflectance_nir_d, half *reflectance_red_d, half *pai_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        pai_d[pos] = __float2half(10.1f * (__half2float(reflectance_nir_d[pos]) - sqrt(__half2float(reflectance_red_d[pos]))) + 3.1f);

        if (__half2float(pai_d[pos]) < 0)
            pai_d[pos] = __float2half(0);
    }
}

__global__ void lai_kernel(half *reflectance_nir_d, half *reflectance_red_d, half *lai_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        half savi = __hdiv(__hmul(__float2half(1.5), __hsub(reflectance_nir_d[pos], reflectance_red_d[pos])), __hadd(__float2half(0.5), __hadd(reflectance_nir_d[pos], reflectance_red_d[pos])));

        if (!__hisnan(savi) && __hgt(savi, __float2half(0.687)))
            lai_d[pos] = __float2half(6.0);
        if (!__hisnan(savi) && __hle(savi, __float2half(0.687)))
            lai_d[pos] = __hdiv(__hneg((half)__logf(__hdiv(__hsub(__float2half(0.69), savi), __float2half(0.59)))), __float2half(0.91));
        if (!__hisnan(savi) && __hlt(savi, __float2half(0.1)))
            lai_d[pos] = __float2half(0.0);

        if (__hlt(lai_d[pos], __float2half(0.0)))
            lai_d[pos] = __float2half(0.0);
    }
}

__global__ void enb_kernel(half *lai_d, half *ndvi_d, half *enb_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (__half2float(ndvi_d[pos]) > 0)
            enb_d[pos] = __float2half((__half2float(lai_d[pos]) < 3) ? 0.97f + 0.0033f * __half2float(lai_d[pos]) : 0.98f);
        else if (__half2float(ndvi_d[pos]) < 0)
            enb_d[pos] = __float2half(0.99f);
        else
            enb_d[pos] = __float2half(NAN);
    }
}

__global__ void eo_kernel(half *lai_d, half *ndvi_d, half *eo_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (__half2float(ndvi_d[pos]) > 0)
            eo_d[pos] = __float2half((__half2float(lai_d[pos]) < 3) ? 0.95f + 0.01f * __half2float(lai_d[pos]) : 0.98f);
        else if (__half2float(ndvi_d[pos]) < 0)
            eo_d[pos] = __float2half(0.985f);
        else
            eo_d[pos] = __float2half(NAN);
    }
}

__global__ void ea_kernel(half *tal_d, half *ea_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        ea_d[pos] = 0.85 * pow((-1 * logf(tal_d[pos])), 0.09);
    }
}

__global__ void surface_temperature_kernel(half *enb_d, half *radiance_termal_d, half *surface_temperature_d, float k1, float k2)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        surface_temperature_d[pos] = __float2half(k2 / (logf((__half2float(enb_d[pos]) * k1 / __half2float(radiance_termal_d[pos])) + 1)));

        if (__half2float(surface_temperature_d[pos]) < 0)
            surface_temperature_d[pos] = __float2half(0);
    }
}

__global__ void short_wave_radiation_kernel(half *tal_d, half *short_wave_radiation_d, float sun_elevation, float distance_earth_sun, float pi)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        short_wave_radiation_d[pos] = __float2half((1367 * sinf(sun_elevation * pi / 180) * __half2float(tal_d[pos])) / (distance_earth_sun * distance_earth_sun));
    }
}

__global__ void large_wave_radiation_surface_kernel(half *surface_temperature_d, half *eo_d, half *large_wave_radiation_surface_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        float temperature_pixel = surface_temperature_d[pos];
        float surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
        large_wave_radiation_surface_d[pos] = __float2half(__half2float(eo_d[pos]) * 5.67 * 1e-8 * surface_temperature_pow_4);
    }
}

__global__ void large_wave_radiation_atmosphere_kernel(half *ea_d, half *large_wave_radiation_atmosphere_d, float temperature)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float temperature_kelvin_pow_4 = temperature * temperature * temperature * temperature;
    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        large_wave_radiation_atmosphere_d[pos] = __half2float(ea_d[pos]) * 5.67 * 1e-8 * temperature_kelvin_pow_4;
    }
}

__global__ void net_radiation_kernel(half *short_wave_radiation_d, half *albedo_d, half *large_wave_radiation_atmosphere_d, half *large_wave_radiation_surface_d, half *eo_d, half *net_radiation_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        net_radiation_d[pos] = __float2half((1.0f - __half2float(albedo_d[pos])) * __half2float(short_wave_radiation_d[pos]) + __half2float(large_wave_radiation_atmosphere_d[pos]) - __half2float(large_wave_radiation_surface_d[pos]) - (1.0f - __half2float(eo_d[pos])) * __half2float(large_wave_radiation_atmosphere_d[pos]));

        if (__half2float(net_radiation_d[pos]) < 0.0f)
            net_radiation_d[pos] = __float2half(0.0f);
    }
}

__global__ void soil_heat_kernel(half *ndvi_d, half *albedo_d, half *surface_temperature_d, half *net_radiation_d, half *soil_heat_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (__half2float(ndvi_d[pos]) >= 0) {
            float temperature_celcius = __half2float(surface_temperature_d[pos]) - 273.15;
            float ndvi_pixel_pow_4 = __half2float(ndvi_d[pos]) * __half2float(ndvi_d[pos]) * __half2float(ndvi_d[pos]) * __half2float(ndvi_d[pos]);
            soil_heat_d[pos] = __float2half(temperature_celcius * (0.0038f + 0.0074f * __half2float(albedo_d[pos])) * (1 - 0.98f * ndvi_pixel_pow_4) * __half2float(net_radiation_d[pos]));
        } else
            soil_heat_d[pos] = __float2half(0.5f * __half2float(net_radiation_d[pos]));

        if (__half2float(soil_heat_d[pos]) < 0)
            soil_heat_d[pos] = __float2half(0);
    }
}
