#include "kernels.cuh"

__device__ int width_d;
__device__ int height_d;

__global__ void rad_kernel(float *band_d, float *radiance_d, float *rad_add_d, float *rad_mult_d, int band_idx)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        radiance_d[pos] = band_d[pos] * rad_mult_d[band_idx] + rad_add_d[band_idx];

        if (radiance_d[pos] <= 0)
            radiance_d[pos] = NAN;
    }
}

__global__ void ref_kernel(float *band_d, float *reflectance_d, float *ref_add_d, float *ref_mult_d, float sin_sun, int band_idx)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        reflectance_d[pos] = (band_d[pos] * ref_mult_d[band_idx] + ref_add_d[band_idx]) / sin_sun;

        if (reflectance_d[pos] <= 0)
            reflectance_d[pos] = NAN;
    }
}

__global__ void albedo_kernel(float *reflectance_blue_d, float *reflectance_green_d, float *reflectance_red_d, float *reflectance_nir_d, float *reflectance_swir1_d, float *reflectance_swir2_d, float *tal_d, float *albedo_d, float *ref_w_coeff_d)
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

        albedo_d[pos] = (alb_toa - 0.03) / (tal_d[pos] * tal_d[pos]);

        if (albedo_d[pos] <= 0)
            albedo_d[pos] = NAN;
    }
}

__global__ void ndvi_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *ndvi_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        ndvi_d[pos] = (reflectance_nir_d[pos] - reflectance_red_d[pos]) / (reflectance_nir_d[pos] + reflectance_red_d[pos]);

        if (ndvi_d[pos] <= -1 || ndvi_d[pos] >= 1)
            ndvi_d[pos] = NAN;
    }
}

__global__ void pai_kernel(float *reflectance_nir_d, float *reflectance_red_d, float *pai_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        pai_d[pos] = 10.1 * (reflectance_nir_d[pos] - sqrt(reflectance_red_d[pos])) + 3.1;

        if (pai_d[pos] < 0)
            pai_d[pos] = 0;
    }
}

__global__ void lai_kernel(float *savi_d, float *lai_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (!isnan(savi_d[pos]) && savi_d[pos] > 0.687)
            lai_d[pos] = 6;
        if (!isnan(savi_d[pos]) && savi_d[pos] < 0.1)
            lai_d[pos] = 0;

        if (lai_d[pos] < 0)
            lai_d[pos] = 0;
    }
}

__global__ void enb_kernel(float *lai_d, float *ndvi_d, float *enb_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (ndvi_d[pos] > 0)
            enb_d[pos] = (lai_d[pos] < 3) ? 0.97 + 0.0033 * lai_d[pos] : 0.98;            
        else if (ndvi_d[pos] < 0)
            enb_d[pos] = 0.99;
        else
            enb_d[pos] = NAN;
    }
}

__global__ void eo_kernel(float *lai_d, float *ndvi_d, float *eo_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (ndvi_d[pos] > 0)
            eo_d[pos] = (lai_d[pos] < 3) ? 0.95 + 0.01 * lai_d[pos] : 0.98;            
        else if (ndvi_d[pos] < 0)
            eo_d[pos] = 0.985;
        else
            eo_d[pos] = NAN;
    }
}

__global__ void surface_temperature_kernel(float *enb_d, float *radiance_termal_d, float *surface_temperature_d, float k1, float k2)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        surface_temperature_d[pos] = k2 / (logf((enb_d[pos] * k1 / radiance_termal_d[pos]) + 1));

        if (surface_temperature_d[pos] < 0)
            surface_temperature_d[pos] = 0;
    }
}

__global__ void short_wave_radiation_kernel(float *tal_d, float *short_wave_radiation_d, float sun_elevation, float distance_earth_sun, float pi)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        short_wave_radiation_d[pos] = (1367 * sin(sun_elevation * pi / 180) * tal_d[pos]) / (distance_earth_sun * distance_earth_sun);
    }
}

__global__ void large_wave_radiation_surface_kernel(float *surface_temperature_d, float *eo_d, float *large_wave_radiation_surface_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        float temperature_pixel = surface_temperature_d[pos];
        float surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
        large_wave_radiation_surface_d[pos] = eo_d[pos] * 5.67 * 1e-8 * surface_temperature_pow_4;
    }
}

__global__ void net_radiation_kernel(float *short_wave_radiation_d, float *albedo_d, float *large_wave_radiation_atmosphere_d, float *large_wave_radiation_surface_d, float *eo_d, float *net_radiation_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;
        net_radiation_d[pos] = (1 - albedo_d[pos]) * short_wave_radiation_d[pos] + large_wave_radiation_atmosphere_d[pos] - large_wave_radiation_surface_d[pos] - (1 - eo_d[pos]) * large_wave_radiation_atmosphere_d[pos];

        if (net_radiation_d[pos] < 0)
            net_radiation_d[pos] = 0;
    }
}

__global__ void soil_heat_kernel(float *ndvi_d, float *albedo_d, float *surface_temperature_d, float *net_radiation_d, float *soil_heat_d)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width_d * height_d) {
        unsigned int row = idx / width_d;
        unsigned int col = idx % width_d;
        unsigned int pos = row * width_d + col;

        if (ndvi_d[pos] >= 0) {
            float temperature_celcius = surface_temperature_d[pos] - 273.15;
            float ndvi_pixel_pow_4 = ndvi_d[pos] * ndvi_d[pos] * ndvi_d[pos] * ndvi_d[pos];
            soil_heat_d[pos] = temperature_celcius * (0.0038 + 0.0074 * albedo_d[pos]) * (1 - 0.98 * ndvi_pixel_pow_4) * net_radiation_d[pos];
        } else
            soil_heat_d[pos] = 0.5 * net_radiation_d[pos];

        if (soil_heat_d[pos] < 0)
            soil_heat_d[pos] = 0;
    }
}
