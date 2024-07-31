#pragma once

#include "constants.h"

/**
 * @brief  Determines if a and b are approximately equals based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if they are approximately equals, and FALSE otherwise.
 */
bool approximatelyEqual(float a, float b);

/**
 * @brief  Determines if a and b are essentially equals based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if they are essentially equals, and FALSE otherwise.
 */
bool essentiallyEqual(float a, float b);

/**
 * @brief  Determines if a is definitely greater than b based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if a is definitely greater than b, and FALSE otherwise.
 */
bool definitelyGreaterThan(float a, float b);

/**
 * @brief  Determines if a is definitely less than b based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if a is definitely less than b, and FALSE otherwise.
 */
bool definitelyLessThan(float a, float b);

/**
 * @brief  Prints a pointer.
 * 
 * @param pointer: Pointer to be printed.
 * @param height: Height of the pointer.
 * @param width: Width of the pointer.
 */
void printLinearPointer(float *pointer, int height, int width);

/**
 * @brief  Saves a TIFF file.
 * 
 * @param path: Path to save the TIFF file.
 * @param data: Data to be saved.
 * @param height: Height of the data.
 * @param width: Width of the data.
 */
void saveTiff(string path, float *data, int height, int width);
