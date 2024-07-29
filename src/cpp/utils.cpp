#include "utils.h"

bool approximatelyEqual(float a, float b)
{
  return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPS);
}

bool essentiallyEqual(float a, float b)
{
  return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * EPS);
}

bool definitelyGreaterThan(float a, float b)
{
  return (a - b) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPS);
}

bool definitelyLessThan(float a, float b)
{
  return (b - a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPS);
}

void printLinearPointer(float *pointer, int height, int width)
{
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      cout << pointer[i * width + j] << " ";
    }
    cout << endl;
  }
}