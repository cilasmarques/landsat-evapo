#include "sensors.h"
#include "surfaceData.h"

string latent_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.latent_heat_flux[i] = products.net_radiation[i] - products.soil_heat[i] - products.sensible_heat_flux[i];
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,LATENT_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string net_radiation_24h_function(Products products, Station station, MTL mtl)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    float FL = 110.0f;
    float dr = (1.0f / mtl.distance_earth_sun) * (1.0f / mtl.distance_earth_sun);
    float sigma = 0.409f * sinf(((2.0f * PI / 365.0f) * mtl.julian_day) - 1.39f);
    float phi = (PI / 180.0f) * station.latitude;
    float omegas = acosf(-tanf(phi) * tanf(sigma));
    float Ra24h = (((24.0f * 60.0f / PI) * GSC * dr) * (omegas * sinf(phi) * sinf(sigma) + cosf(phi) * cosf(sigma) * sinf(omegas))) * (1000000.0f / 86400.0f);
    float Rs24h = station.INTERNALIZATION_FACTOR * sqrtf(station.v7_max - station.v7_min) * Ra24h;

    for (int i = 0; i < products.height_band * products.width_band; i++) {
        products.net_radiation_24h[i] = (1.0f - products.albedo[i]) * Rs24h - FL * Rs24h / Ra24h;
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,NET_RADIATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string evapotranspiration_24h_function(Products products, Station station)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    for (int i = 0; i < products.height_band * products.width_band; i++) {
        float temperature_celcius = products.surface_temperature[i] - 273.15f;
        products.evapotranspiration_24h[i] = (86400.0f / ((2.501f - 0.0236f * temperature_celcius) * powf(10.0f, 6.0f))) * (products.latent_heat_flux[i] / (products.net_radiation[i] - products.soil_heat[i])) * products.net_radiation_24h[i];
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "SERIAL,EVAPOTRANSPIRATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::compute_H_ET(Products products, Station station, MTL mtl)
{
    string result = "";
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    result += latent_heat_flux_function(products);
    result += net_radiation_24h_function(products, station, mtl);
    result += evapotranspiration_24h_function(products, station);
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "SERIAL,P4_FINAL_PROD," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};
