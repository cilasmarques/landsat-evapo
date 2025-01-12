#include "constants.h"
#include "surfaceData.h"

Endmember::Endmember()
{
    this->ndvi = 0;
    this->temperature = 0;
    this->line = 0;
    this->col = 0;
}

Endmember::Endmember(float ndvi, float temperature, int line, int col)
{
    this->ndvi = ndvi;
    this->temperature = temperature;
    this->line = line;
    this->col = col;
}

void get_quartiles(float *target, float *v_quartile, int height_band, int width_band, float first_interval, float middle_interval, float last_interval)
{
    const int SIZE = height_band * width_band;
    float *target_values = (float *)malloc(sizeof(float) * SIZE);

    if (target_values == NULL)
        exit(15);

    int pos = 0;
    for (int i = 0; i < height_band * width_band; i++) {
        if (!isnan(target[i]) && !isinf(target[i])) {
            target_values[pos] = target[i];
            pos++;
        }
    }

    int first_index = static_cast<int>(floor(first_interval * pos));
    int middle_index = static_cast<int>(floor(middle_interval * pos));
    int last_index = static_cast<int>(floor(last_interval * pos));

    std::nth_element(target_values, target_values + first_index, target_values + pos);
    v_quartile[0] = target_values[first_index];

    std::nth_element(target_values, target_values + middle_index, target_values + pos);
    v_quartile[1] = target_values[middle_index];

    std::nth_element(target_values, target_values + last_index, target_values + pos);
    v_quartile[2] = target_values[last_index];

    free(target_values);
}

string getEndmembersSTEEP(Products products)
{
    string result = "";
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    vector<Endmember> hotCandidates;
    vector<Endmember> coldCandidates;

    vector<float> tsQuartile(3);
    vector<float> ndviQuartile(3);
    vector<float> albedoQuartile(3);
    try {
        initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

        begin = system_clock::now();
        get_quartiles(products.ndvi, ndviQuartile.data(), products.height_band, products.width_band, 0.15, 0.97, 0.97);
        get_quartiles(products.albedo, albedoQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75);
        get_quartiles(products.surface_temperature, tsQuartile.data(), products.height_band, products.width_band, 0.20, 0.85, 0.97);

        for (int i = 0; i < products.height_band * products.width_band; i++) {
            bool hotNDVI = !std::isnan(products.ndvi[i]) && products.ndvi[i] > 0.10 && products.ndvi[i] < ndviQuartile[0];
            bool hotAlbedo = !std::isnan(products.albedo[i]) && products.albedo[i] > albedoQuartile[1] && products.albedo[i] < albedoQuartile[2];
            bool hotTS = !std::isnan(products.surface_temperature[i]) && products.surface_temperature[i] > tsQuartile[1] && products.surface_temperature[i] < tsQuartile[2];

            bool coldNDVI = !std::isnan(products.ndvi[i]) && products.ndvi[i] > ndviQuartile[2];
            bool coldAlbedo = !std::isnan(products.albedo[i]) && products.albedo[i] > albedoQuartile[0] && products.albedo[i] < albedoQuartile[1];
            bool coldTS = !std::isnan(products.surface_temperature[i]) && products.surface_temperature[i] < tsQuartile[0];

            int line = i / products.width_band;
            int col = i % products.width_band;

            if (hotAlbedo && hotNDVI && hotTS) 
                hotCandidates.emplace_back(products.ndvi[i], products.surface_temperature[i], line, col);

            if (coldNDVI && coldAlbedo && coldTS) 
                coldCandidates.emplace_back(products.ndvi[i], products.surface_temperature[i], line, col);
        }
        end = system_clock::now();

        int hot_pos = static_cast<unsigned int>(std::floor(hotCandidates.size() * 0.5));
        int cold_pos = static_cast<unsigned int>(std::floor(coldCandidates.size() * 0.5));

        if (hotCandidates.size() == 0)
            throw std::runtime_error("No hot candidates found");
        if (coldCandidates.size() == 0)
            throw std::runtime_error("No cold candidates found");

        std::sort(hotCandidates.begin(), hotCandidates.end(), CompareEndmemberTemperature());
        std::sort(coldCandidates.begin(), coldCandidates.end(), CompareEndmemberTemperature());

        int hotIndexes[2] = {hotCandidates[hot_pos].line, hotCandidates[hot_pos].col};
        int coldIndexes[2] = {coldCandidates[cold_pos].line, coldCandidates[cold_pos].col};

        memcpy(products.hotEndmemberPos, hotIndexes, sizeof(int) * 2);
        memcpy(products.coldEndmemberPos, coldIndexes, sizeof(int) * 2);

        end = system_clock::now();
        general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
        final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
        result += "SERIAL,PIXEL_FILTER," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    } catch (const std::exception &e) {
        cerr << "Pixel filtering error: " << e.what() << endl;
        exit(15);
    }

    return result;
}

string getEndmembersASEBAL(Products products)
{
    string result = "";
    system_clock::time_point begin, end;
    int64_t initial_time, final_time;
    float general_time;

    vector<Endmember> hotCandidates;
    vector<Endmember> coldCandidates;

    vector<float> tsQuartile(3);
    vector<float> ndviQuartile(3);
    vector<float> albedoQuartile(3);
    try {
        initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

        begin = system_clock::now();
        get_quartiles(products.ndvi, ndviQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75);
        get_quartiles(products.albedo, albedoQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75);
        get_quartiles(products.surface_temperature, tsQuartile.data(), products.height_band, products.width_band, 0.25, 0.50, 0.75);

        for (int i = 0; i < products.height_band * products.width_band; i++) {
            bool hotNDVI = !isnan(products.ndvi[i]) && products.ndvi[i] > 0.10 && products.ndvi[i] < ndviQuartile[0];
            bool hotAlbedo = !isnan(products.albedo[i]) && products.albedo[i] > albedoQuartile[2];
            bool hotTS = !isnan(products.surface_temperature[i]) && products.surface_temperature[i] > tsQuartile[2];

            bool coldNDVI = !isnan(products.ndvi[i]) && products.ndvi[i] > ndviQuartile[2];
            bool coldAlbedo = !isnan(products.albedo[i]) && products.albedo[i] < albedoQuartile[1];
            bool coldTS = !isnan(products.surface_temperature[i]) && products.surface_temperature[i] < tsQuartile[0];

            int line = i / products.width_band;
            int col = i % products.width_band;

            if (hotAlbedo && hotNDVI && hotTS) 
                hotCandidates.emplace_back(products.ndvi[i], products.surface_temperature[i], line, col);

            if (coldNDVI && coldAlbedo && coldTS) 
                coldCandidates.emplace_back(products.ndvi[i], products.surface_temperature[i], line, col);
        }
        end = system_clock::now();

        int hot_pos = static_cast<unsigned int>(std::floor(hotCandidates.size() * 0.5));
        int cold_pos = static_cast<unsigned int>(std::floor(coldCandidates.size() * 0.5));

        if (hotCandidates.size() == 0)
            throw std::runtime_error("No hot candidates found");
        if (coldCandidates.size() == 0)
            throw std::runtime_error("No cold candidates found");

        std::sort(hotCandidates.begin(), hotCandidates.end(), CompareEndmemberTemperature());
        std::sort(coldCandidates.begin(), coldCandidates.end(), CompareEndmemberTemperature());

        int hotIndexes[2] = {hotCandidates[hot_pos].line, hotCandidates[hot_pos].col};
        int coldIndexes[2] = {coldCandidates[cold_pos].line, coldCandidates[cold_pos].col};

        memcpy(products.hotEndmemberPos, hotIndexes, sizeof(int) * 2);
        memcpy(products.coldEndmemberPos, coldIndexes, sizeof(int) * 2);

        end = system_clock::now();
        general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
        final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
        result += "SERIAL,PIXEL_FILTER," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    } catch (const std::exception &e) {
        cerr << "Pixel filtering error: " << e.what() << endl;
        exit(15);
    }

    return result;
}

string Products::select_endmembers(Products products)
{
    string result = "";
    int64_t initial_time, final_time;
    system_clock::time_point begin, end;
    float general_time;

    initial_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

    begin = system_clock::now();
    if (model_method == 0) { // STEEP
        result += getEndmembersSTEEP(products);
    } else if (model_method == 1) { // ASEBAL
        result += getEndmembersASEBAL(products);
    }
    end = system_clock::now();

    general_time = duration_cast<nanoseconds>(end - begin).count() / 1000000.0;
    final_time = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    result += "SERIAL,P2_PIXEL_SEL," + std::to_string(general_time) + "," + std::to_string(initial_time) + "," + std::to_string(final_time) + "\n";
    return result;
}
