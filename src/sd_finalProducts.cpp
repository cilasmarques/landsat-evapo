#include "sensors.h"
#include "surfaceData.h"
#include "thread_utils.h" // Include for parallel_for

// Helper function to process a range of rows for latent_heat_flux_function
void latent_heat_flux_function_worker(Products products, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            products.latent_heat_flux[i] = products.net_radiation[i] - products.soil_heat[i] - products.sensible_heat_flux[i];
        }
    }
}

string latent_heat_flux_function(Products products)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    parallel_for(products.height_band, [&](int start_row, int end_row) {
        latent_heat_flux_function_worker(products, start_row, end_row);
    });

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,LATENT_HEAT_FLUX," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

// Helper function to process a range of rows for net_radiation_24h_function
void net_radiation_24h_function_worker(Products products, float Rs24h_val, float Ra24h_val, float FL_val, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            products.net_radiation_24h[i] = (1.0 - products.albedo[i]) * Rs24h_val - FL_val * Rs24h_val / Ra24h_val;
        }
    }
}

string net_radiation_24h_function(Products products, Station station, MTL mtl)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    float FL = 110.0;
    float dr = (1.0 / mtl.distance_earth_sun) * (1.0 / mtl.distance_earth_sun);
    float sigma = 0.409 * sin(((2.0 * PI / 365.0) * mtl.julian_day) - 1.39);
    float phi = (PI / 180.0) * station.latitude;
    float omegas = acos(-tan(phi) * tan(sigma));
    float Ra24h = (((24.0 * 60.0 / PI) * GSC * dr) * (omegas * sin(phi) * sin(sigma) + cos(phi) * cos(sigma) * sin(omegas))) * (1000000.0 / 86400.0);
    float Rs24h = station.INTERNALIZATION_FACTOR * sqrt(station.v7_max - station.v7_min) * Ra24h;

    parallel_for(products.height_band, [&](int start_row, int end_row) {
        net_radiation_24h_function_worker(products, Rs24h, Ra24h, FL, start_row, end_row);
    });

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,NET_RADIATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

// Helper function to process a range of rows for evapotranspiration_24h_function
void evapotranspiration_24h_function_worker(Products products, int start_row, int end_row) {
    for (int r = start_row; r < end_row; ++r) {
        for (int c = 0; c < products.width_band; ++c) {
            int i = r * products.width_band + c;
            float temperature_celcius = products.surface_temperature[i] - 273.15;
            products.evapotranspiration_24h[i] = (86400.0 / ((2.501 - 0.0236 * temperature_celcius) * pow(10.0, 6.0))) * (products.latent_heat_flux[i] / (products.net_radiation[i] - products.soil_heat[i])) * products.net_radiation_24h[i];
        }
    }
}

string evapotranspiration_24h_function(Products products, Station station)
{
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    begin = system_clock::now();

    parallel_for(products.height_band, [&](int start_row, int end_row) {
        evapotranspiration_24h_function_worker(products, start_row, end_row);
    });

    end = system_clock::now();
    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return "PARALLEL,EVAPOTRANSPIRATION_24H," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
};

string Products::compute_H_ET(Products products, Station station, MTL mtl)
{
    string result = "";
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    // These functions are now internally parallelized
    result += latent_heat_flux_function(products);
    result += net_radiation_24h_function(products, station, mtl);
    result += evapotranspiration_24h_function(products, station);
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "PARALLEL,P4_FINAL_PROD," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
};