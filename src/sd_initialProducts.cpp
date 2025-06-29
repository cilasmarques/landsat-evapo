#include "constants.h"
#include "sensors.h"
#include "surfaceData.h"

string radiance_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    // float *bands[] = {products.band_blue, products.band_green, products.band_red, products.band_nir, products.band_swir1, products.band_termal, products.band_swir2};
    // float *radiances[] = {products.radiance_blue, products.radiance_green, products.radiance_red, products.radiance_nir, products.radiance_swir1, products.radiance_termal, products.radiance_swir2};
    // int band_ids[] = {PARAM_BAND_BLUE_INDEX, PARAM_BAND_GREEN_INDEX, PARAM_BAND_RED_INDEX, PARAM_BAND_NIR_INDEX, PARAM_BAND_SWIR1_INDEX, PARAM_BAND_TERMAL_INDEX, PARAM_BAND_SWIR2_INDEX};

    // for (int b = 0; b < 7; b++) {
    //     for (int i = 0; i < products.height_band * products.width_band; i++) {
    //         radiances[b][i] = bands[b][i] * mtl.rad_mult[band_ids[b]] + mtl.rad_add[band_ids[b]];
    //         if (radiances[b][i] <= 0)
    //             radiances[b][i] = NAN;
    //     }
    // }

    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.radiance_termal[i] = products.band_termal[i] * mtl.rad_mult[PARAM_BAND_TERMAL_INDEX] + mtl.rad_add[PARAM_BAND_TERMAL_INDEX];
        if (products.radiance_termal[i] <= 0)
            products.radiance_termal[i] = NAN;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,RADIANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string reflectance_function(Products products, MTL mtl)
{
    const float sin_sun = sinf(mtl.sun_elevation * PI / 180.0f);

    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    float *bands[] = {products.band_blue, products.band_green, products.band_red, products.band_nir, products.band_swir1, products.band_termal, products.band_swir2};
    float *reflectances[] = {products.reflectance_blue, products.reflectance_green, products.reflectance_red, products.reflectance_nir, products.reflectance_swir1, products.reflectance_termal, products.reflectance_swir2};
    int indices[] = {PARAM_BAND_BLUE_INDEX, PARAM_BAND_GREEN_INDEX, PARAM_BAND_RED_INDEX, PARAM_BAND_NIR_INDEX, PARAM_BAND_SWIR1_INDEX, PARAM_BAND_TERMAL_INDEX, PARAM_BAND_SWIR2_INDEX};

    for (int b = 0; b < 7; b++) {
        for (int i = 0; i < products.height_band * products.width_band; i++) {
            reflectances[b][i] = (bands[b][i] * mtl.ref_mult[indices[b]] + mtl.ref_add[indices[b]]) / sin_sun;
            if (reflectances[b][i] <= 0)
                reflectances[b][i] = NAN;
        }
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,REFLECTANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string albedo_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float alb = products.reflectance_blue[i] * mtl.ref_w_coeff[PARAM_BAND_BLUE_INDEX] +
                    products.reflectance_green[i] * mtl.ref_w_coeff[PARAM_BAND_GREEN_INDEX] +
                    products.reflectance_red[i] * mtl.ref_w_coeff[PARAM_BAND_RED_INDEX] +
                    products.reflectance_nir[i] * mtl.ref_w_coeff[PARAM_BAND_NIR_INDEX] +
                    products.reflectance_swir1[i] * mtl.ref_w_coeff[PARAM_BAND_SWIR1_INDEX] +
                    products.reflectance_swir2[i] * mtl.ref_w_coeff[PARAM_BAND_SWIR2_INDEX];

        products.albedo[i] = (alb - 0.03f) / (products.tal[i] * products.tal[i]);

        if (products.albedo[i] <= 0)
            products.albedo[i] = NAN;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,ALBEDO," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string ndvi_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.ndvi[i] = (products.reflectance_nir[i] - products.reflectance_red[i]) / (products.reflectance_nir[i] + products.reflectance_red[i]);

        if (products.ndvi[i] <= -1 || products.ndvi[i] >= 1)
            products.ndvi[i] = NAN;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,NDVI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string pai_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.pai[i] = 10.1f * (products.reflectance_nir[i] - sqrtf(products.reflectance_red[i])) + 3.1f;

        if (products.pai[i] < 0)
            products.pai[i] = 0;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,PAI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string lai_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float savi = ((1.5f) * (products.reflectance_nir[i] - products.reflectance_red[i])) / (0.5f + (products.reflectance_nir[i] + products.reflectance_red[i]));

        if (!isnan(savi) && savi > 0.687f)
            products.lai[i] = 6.0f;
        if (!isnan(savi) && savi <= 0.687f)
            products.lai[i] = -logf((0.69f - savi) / 0.59f) / 0.91f;
        if (!isnan(savi) && savi < 0.1f)
            products.lai[i] = 0.0f;

        if (products.lai[i] < 0.0f)
            products.lai[i] = 0.0f;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,LAI," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string enb_emissivity_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        if (products.ndvi[i] > 0)
            products.enb_emissivity[i] = (products.lai[i] < 3.0f) ? 0.97f + 0.0033f * products.lai[i] : 0.98f;
        else if (products.ndvi[i] < 0)
            products.enb_emissivity[i] = 0.99f;
        else
            products.enb_emissivity[i] = NAN;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,ENB_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string eo_emissivity_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        if (products.ndvi[i] > 0)
            products.eo_emissivity[i] = (products.lai[i] < 3.0f) ? 0.95f + 0.01f * products.lai[i] : 0.98f;
        else if (products.ndvi[i] < 0)
            products.eo_emissivity[i] = 0.985f;
        else
            products.eo_emissivity[i] = NAN;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,EO_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string ea_emissivity_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++)
        products.ea_emissivity[i] = 0.85f * powf((-1.0f * logf(products.tal[i])), 0.09f);
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,EA_EMISSIVITY," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string surface_temperature_function(Products products, MTL mtl)
{
    float k1, k2;
    switch (mtl.number_sensor) {
    case 5:
        k1 = 607.76;
        k2 = 1260.56;
        break;

    case 7:
        k1 = 666.09;
        k2 = 1282.71;
        break;

    case 8:
        k1 = 774.8853;
        k2 = 1321.0789;
        break;

    default:
        cerr << "Sensor problem!";
        exit(6);
    }

    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.surface_temperature[i] = k2 / (logf((products.enb_emissivity[i] * k1 / products.radiance_termal[i]) + 1));

        if (products.surface_temperature[i] < 0.0f)
            products.surface_temperature[i] = 0.0f;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,SURFACE_TEMPERATURE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string short_wave_radiation_function(Products products, MTL mtl)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++)
        products.short_wave_radiation[i] = (1367.0f * sinf(mtl.sun_elevation * PI / 180.0f) * products.tal[i]) / (mtl.distance_earth_sun * mtl.distance_earth_sun);
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,SHORT_WAVE_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string large_waves_radiations_function(Products products, float temperature)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float temperature_pixel = products.surface_temperature[i];
        float surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
        products.large_wave_radiation_surface[i] = products.eo_emissivity[i] * 5.67f * 1e-8f * surface_temperature_pow_4;

        float station_temperature_kelvin_pow_4 = temperature * temperature * temperature * temperature;
        products.large_wave_radiation_atmosphere[i] = products.ea_emissivity[i] * 5.67f * 1e-8f * station_temperature_kelvin_pow_4;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,LARGE_WAVES_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.net_radiation[i] = (1.0f - products.albedo[i]) * products.short_wave_radiation[i] + products.large_wave_radiation_atmosphere[i] - products.large_wave_radiation_surface[i] - (1.0f - products.eo_emissivity[i]) * products.large_wave_radiation_atmosphere[i];

        if (products.net_radiation[i] < 0.0f)
            products.net_radiation[i] = 0.0f;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,NET_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string soil_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        if (products.ndvi[i] >= 0) {
            float ndvi_pixel_pow_4 = powf(products.ndvi[i], 4.0f);
            float temperature_celcius = products.surface_temperature[i] - 273.15f;
            products.soil_heat[i] = temperature_celcius * (0.0038f + 0.0074f * products.albedo[i]) * (1.0f - 0.98f * ndvi_pixel_pow_4) * products.net_radiation[i];
        } else
            products.soil_heat[i] = 0.5f * products.net_radiation[i];

        if (products.soil_heat[i] < 0.0f)
            products.soil_heat[i] = 0.0f;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,SOIL_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::compute_Rn_G(Products products, Station station, MTL mtl)
{
    string result = "";
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    result += radiance_function(products, mtl);
    result += reflectance_function(products, mtl);
    result += albedo_function(products, mtl);

    // Vegetation indices
    result += ndvi_function(products);
    result += lai_function(products);
    if (model_method == 0)  
        result += pai_function(products);

    // Emissivity indices
    result += enb_emissivity_function(products);
    result += eo_emissivity_function(products);
    result += ea_emissivity_function(products);
    result += surface_temperature_function(products, mtl);

    // Radiation waves
    result += short_wave_radiation_function(products, mtl);
    result += large_waves_radiations_function(products, station.temperature_image);

    // Main products
    result += net_radiation_function(products);
    result += soil_heat_flux_function(products);
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "SERIAL,P1_INITIAL_PROD," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
}
