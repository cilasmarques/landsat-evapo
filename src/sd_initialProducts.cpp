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
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.radiance_blue[i] = products.band_blue[i] * mtl.rad_mult[PARAM_BAND_BLUE_INDEX] + mtl.rad_add[PARAM_BAND_BLUE_INDEX];
        products.radiance_green[i] = products.band_green[i] * mtl.rad_mult[PARAM_BAND_GREEN_INDEX] + mtl.rad_add[PARAM_BAND_GREEN_INDEX];
        products.radiance_red[i] = products.band_red[i] * mtl.rad_mult[PARAM_BAND_RED_INDEX] + mtl.rad_add[PARAM_BAND_RED_INDEX];
        products.radiance_nir[i] = products.band_nir[i] * mtl.rad_mult[PARAM_BAND_NIR_INDEX] + mtl.rad_add[PARAM_BAND_NIR_INDEX];
        products.radiance_swir1[i] = products.band_swir1[i] * mtl.rad_mult[PARAM_BAND_SWIR1_INDEX] + mtl.rad_add[PARAM_BAND_SWIR1_INDEX];
        products.radiance_termal[i] = products.band_termal[i] * mtl.rad_mult[PARAM_BAND_TERMAL_INDEX] + mtl.rad_add[PARAM_BAND_TERMAL_INDEX];
        products.radiance_swir2[i] = products.band_swir2[i] * mtl.rad_mult[PARAM_BAND_SWIR2_INDEX] + mtl.rad_add[PARAM_BAND_SWIR2_INDEX];

        if (products.radiance_blue[i] <= 0)
            products.radiance_blue[i] = NAN;
        if (products.radiance_green[i] <= 0)
            products.radiance_green[i] = NAN;
        if (products.radiance_red[i] <= 0)
            products.radiance_red[i] = NAN;
        if (products.radiance_nir[i] <= 0)
            products.radiance_nir[i] = NAN;
        if (products.radiance_swir1[i] <= 0)
            products.radiance_swir1[i] = NAN;
        if (products.radiance_termal[i] <= 0)
            products.radiance_termal[i] = NAN;
        if (products.radiance_swir2[i] <= 0)
            products.radiance_swir2[i] = NAN;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,RADIANCE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
}

string reflectance_function(Products products, MTL mtl)
{
    const float sin_sun = sin(mtl.sun_elevation * PI / 180);

    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.reflectance_blue[i] = (products.band_blue[i] * mtl.ref_mult[PARAM_BAND_BLUE_INDEX] + mtl.ref_add[PARAM_BAND_BLUE_INDEX]) / sin_sun;
        products.reflectance_green[i] = (products.band_green[i] * mtl.ref_mult[PARAM_BAND_GREEN_INDEX] + mtl.ref_add[PARAM_BAND_GREEN_INDEX]) / sin_sun;
        products.reflectance_red[i] = (products.band_red[i] * mtl.ref_mult[PARAM_BAND_RED_INDEX] + mtl.ref_add[PARAM_BAND_RED_INDEX]) / sin_sun;
        products.reflectance_nir[i] = (products.band_nir[i] * mtl.ref_mult[PARAM_BAND_NIR_INDEX] + mtl.ref_add[PARAM_BAND_NIR_INDEX]) / sin_sun;
        products.reflectance_swir1[i] = (products.band_swir1[i] * mtl.ref_mult[PARAM_BAND_SWIR1_INDEX] + mtl.ref_add[PARAM_BAND_SWIR1_INDEX]) / sin_sun;
        products.reflectance_termal[i] = (products.band_termal[i] * mtl.ref_mult[PARAM_BAND_TERMAL_INDEX] + mtl.ref_add[PARAM_BAND_TERMAL_INDEX]) / sin_sun;
        products.reflectance_swir2[i] = (products.band_swir2[i] * mtl.ref_mult[PARAM_BAND_SWIR2_INDEX] + mtl.ref_add[PARAM_BAND_SWIR2_INDEX]) / sin_sun;

        if (products.reflectance_blue[i] <= 0)
            products.reflectance_blue[i] = NAN;
        if (products.reflectance_green[i] <= 0)
            products.reflectance_green[i] = NAN;
        if (products.reflectance_red[i] <= 0)
            products.reflectance_red[i] = NAN;
        if (products.reflectance_nir[i] <= 0)
            products.reflectance_nir[i] = NAN;
        if (products.reflectance_swir1[i] <= 0)
            products.reflectance_swir1[i] = NAN;
        if (products.reflectance_termal[i] <= 0)
            products.reflectance_termal[i] = NAN;
        if (products.reflectance_swir2[i] <= 0)
            products.reflectance_swir2[i] = NAN;
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

        products.albedo[i] = (alb - 0.03) / (products.tal[i] * products.tal[i]);

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
        products.pai[i] = 10.1 * (products.reflectance_nir[i] - sqrt(products.reflectance_red[i])) + 3.1;

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
        float savi = ((1 + 0.5) * (products.reflectance_nir[i] - products.reflectance_red[i])) / (0.5 + (products.reflectance_nir[i] + products.reflectance_red[i]));

        if (!isnan(savi) && savi > 0.687)
            products.lai[i] = 6;
        if (!isnan(savi) && savi <= 0.687)
            products.lai[i] = -log((0.69 - savi) / 0.59) / 0.91;
        if (!isnan(savi) && savi < 0.1)
            products.lai[i] = 0;

        if (products.lai[i] < 0)
            products.lai[i] = 0;
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
            products.enb_emissivity[i] = (products.lai[i] < 3) ? 0.97 + 0.0033 * products.lai[i] : 0.98;
        else if (products.ndvi[i] < 0)
            products.enb_emissivity[i] = 0.99;
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
            products.eo_emissivity[i] = (products.lai[i] < 3) ? 0.95 + 0.01 * products.lai[i] : 0.98;
        else if (products.ndvi[i] < 0)
            products.eo_emissivity[i] = 0.985;
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
        products.ea_emissivity[i] = 0.85 * pow((-1 * log(products.tal[i])), 0.09);
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

        if (products.surface_temperature[i] < 0)
            products.surface_temperature[i] = 0;
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
        products.short_wave_radiation[i] = (1367 * sin(mtl.sun_elevation * PI / 180) * products.tal[i]) / (mtl.distance_earth_sun * mtl.distance_earth_sun);
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,SHORT_WAVE_RADIATION," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string large_wave_radiation_surface_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float surface_temperature_pow_4 = pow(products.surface_temperature[i], 4);
        products.large_wave_radiation_surface[i] = products.eo_emissivity[i] * 5.67 * 1e-8 * surface_temperature_pow_4;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,LARGE_WAVE_RADIATION_SURFACE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string large_wave_radiation_atmosphere_function(Products products, float temperature)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float surface_temperature_pow_4 = pow(products.surface_temperature[i], 4);
        products.large_wave_radiation_surface[i] = products.eo_emissivity[i] * 5.67 * 1e-8 * surface_temperature_pow_4;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,LARGE_WAVE_RADIATION_ATMOSPHERE," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.net_radiation[i] = (1 - products.albedo[i]) * products.short_wave_radiation[i] + products.large_wave_radiation_atmosphere[i] - products.large_wave_radiation_surface[i] - (1 - products.eo_emissivity[i]) * products.large_wave_radiation_atmosphere[i];

        if (products.net_radiation[i] < 0)
            products.net_radiation[i] = 0;
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
            float ndvi_pixel_pow_4 = pow(products.ndvi[i], 4);
            float temperature_celcius = products.surface_temperature[i] - 273.15;
            products.soil_heat[i] = temperature_celcius * (0.0038 + 0.0074 * products.albedo[i]) * (1 - 0.98 * ndvi_pixel_pow_4) * products.net_radiation[i];
        } else
            products.soil_heat[i] = 0.5 * products.net_radiation[i];

        if (products.soil_heat[i] < 0)
            products.soil_heat[i] = 0;
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
    result += pai_function(products);
    result += lai_function(products);

    // Emissivity indices
    result += enb_emissivity_function(products);
    result += eo_emissivity_function(products);
    result += ea_emissivity_function(products);
    result += surface_temperature_function(products, mtl);

    // Radiation waves
    result += short_wave_radiation_function(products, mtl);
    result += large_wave_radiation_surface_function(products);
    result += large_wave_radiation_atmosphere_function(products, station.temperature_image);

    // Main products
    result += net_radiation_function(products);
    result += soil_heat_flux_function(products);
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "SERIAL,P1_INITIAL_PROD," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
}
