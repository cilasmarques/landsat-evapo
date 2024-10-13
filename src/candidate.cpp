#include "candidate.h"

Candidate::Candidate()
{
  this->ndvi = 0;
  this->temperature = 0;
  this->net_radiation = 0;
  this->soil_heat_flux = 0;
  this->ho = 0;
  this->line = 0;
  this->col = 0;
  this->zom = 0;
  this->ustar = 0;
}

Candidate::Candidate(float ndvi, float temperature, float net_radiation, float soil_heat_flux, float ho, int line, int col)
{
  this->ndvi = ndvi;
  this->temperature = temperature;
  this->net_radiation = net_radiation;
  this->soil_heat_flux = soil_heat_flux;
  this->ho = ho;
  this->line = line;
  this->col = col;
  this->zom = 0;
  this->ustar = 0;
}

void Candidate::setAerodynamicResistance(float newRah)
{
  this->aerodynamic_resistance = newRah;
}

bool equals(Candidate a, Candidate b)
{
  return (a.col == b.col) && (a.line == b.line);
}

bool compare_candidate_temperature(Candidate a, Candidate b)
{
  bool result = a.temperature < b.temperature;

  if (a.temperature == b.temperature)
    result = a.ndvi < b.ndvi;

  return result;
}
