int pulsesPerSlope[NUM_CIRCLES][NUM_SLOPES_PER_QUARTER_CIRCLE];

enum SlopeToIndex
{
    SLOPE_HORIZONTAL = 0,
    SLOPE_EIGHTH = 1,
    SLOPE_QUARTER = 2,
    SLOPE_HALF = 3,
    SLOPE_ONE = 4,
    SLOPE_TWO = 5,
    SLOPE_FOUR = 6,
    SLOPE_EIGHT = 7,
    SLOPE_VERTICAL = 8
};


// Makes a full circle of the given size (0=largest, NUM_CIRCLES-1=smallest) 
// starting from the given quarter (0=top, 1=left, 2=bottom, 3=right).
// calculatePulsesPerSlope() must be called before makeCircle()
void makeCircle(int circle, int firstQuarter)
{
    bool firstPass = true;
    int slopeIndex;

    setScale(xMotor, EIGHTH_STEPS);
    digitalWrite(xMotor[STEP_PIN], LOW);

    for (int quarterCircle = firstQuarter; quarterCircle != firstQuarter || firstPass; quarterCircle = (quarterCircle + 1) % 4)
    {
        firstPass = false;

        // Set Y-direction
        if (quarterCircle == 0 || quarterCircle == 1)
            digitalWrite(yMotor[DIR_PIN], DOWN);
        else
            digitalWrite(yMotor[DIR_PIN], UP);

        // Set X-direction
        if (quarterCircle == 0 || quarterCircle == 3)
            digitalWrite(xMotor[DIR_PIN], LEFT);
        else
            digitalWrite(xMotor[DIR_PIN], RIGHT);

        // Set starting point for the while loop
        if (quarterCircle == 0 || quarterCircle == 2)
            slopeIndex = SLOPE_HORIZONTAL;
        else
            slopeIndex = NUM_SLOPES_PER_QUARTER_CIRCLE - 1;

        // Iterate through pulsesPerSlope
        while (slopeIndex >= SLOPE_HORIZONTAL  &&  slopeIndex <= SLOPE_VERTICAL)
        {
            // Y-scale
            if (slopeIndex == 5)
                setScale(yMotor, QUARTER_STEPS);    
            else if (slopeIndex == 6)
                setScale(yMotor, HALF_STEPS);    
            else if (slopeIndex == 7)
                setScale(yMotor, WHOLE_STEPS);    
            else
                setScale(yMotor, EIGHTH_STEPS);    

            // X-scale
            if (slopeIndex == SLOPE_EIGHTH)
                setScale(xMotor, WHOLE_STEPS);
            else if (slopeIndex == SLOPE_QUARTER)
                setScale(xMotor, HALF_STEPS);
            else if (slopeIndex == SLOPE_HALF)
                setScale(xMotor, QUARTER_STEPS);
            else 
                setScale(xMotor, EIGHTH_STEPS);

            // Send pulses for the current slopeIndex
            // Separate loops for vertical and horizontal cases to optimize performance
            if (slopeIndex == SLOPE_VERTICAL)
            {
                for (int i = 0; i < pulsesPerSlope[circle][slopeIndex]; i++)
                {
                    // Y-Step
                    digitalWrite(yMotor[STEP_PIN], LOW);
                    digitalWrite(yMotor[STEP_PIN], HIGH);
                }
            }
            else if (slopeIndex == SLOPE_HORIZONTAL)
            {
                for (int i = 0; i < pulsesPerSlope[circle][slopeIndex]; i++)
                {
                    // X-Step
                    digitalWrite(xMotor[STEP_PIN], LOW);
                    digitalWrite(xMotor[STEP_PIN], HIGH);
                }
            }
            else
            {
                for (int i = 0; i < pulsesPerSlope[circle][slopeIndex]; i++)
                {
                    // X-Step
                    digitalWrite(xMotor[STEP_PIN], LOW);
                    digitalWrite(xMotor[STEP_PIN], HIGH);

                    // Y-Step
                    digitalWrite(yMotor[STEP_PIN], LOW);
                    digitalWrite(yMotor[STEP_PIN], HIGH);
                }
            }

            // Increment or Decrement slopeIndex
            if (quarterCircle == 0 || quarterCircle == 2)
                slopeIndex++;
            else
                slopeIndex--;
        }
    }
}

// Calculates and stores the number of pulses that need to be made at each slope
// Based on MILLIMETERS_PER_UNITSPACE, STEPS_PER_MILLIMETER, and NUM_CIRCLES 
void calculatePulsesPerSlope(){

    int outerRadius, deltaR;

    // Calculate the radius of the largest circle in eighth steps
    outerRadius = MILLIMETERS_PER_UNITSPACE * STEPS_PER_MILLIMETER * 8;

    // Ensure that each radius is evenly spaced from the center
    outerRadius = outerRadius - outerRadius % NUM_CIRCLES;

    // Calculate the spacing between the circles
    deltaR = outerRadius / NUM_CIRCLES;

    // Loop over each circle
    for (int circle = 0; circle < NUM_CIRCLES; circle++)
    {
        int radius, xStepsRemaining, yStepsRemaining;

        // Radius of the current circle
        radius = outerRadius - deltaR * circle;

        // StepsRemaining records the number of steps that need to be taken to reach the leftmost point on the circle
        // Initialize to the total number of steps per quarter circle. The starting point is the topmost point
        xStepsRemaining = radius;
        yStepsRemaining = radius;

        // Horizontal steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 1.0/16.0)
        {
           pulsesPerSlope[circle][SLOPE_HORIZONTAL]++;
           xStepsRemaining--;
        }

        // Eighth slope steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/16.0  &&  xStepsRemaining >= 8)
        {
            pulsesPerSlope[circle][SLOPE_EIGHTH]++;
            xStepsRemaining -= 8;
            yStepsRemaining--;
        }

        // Quarter slope steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/8.0  &&  xStepsRemaining >= 4)
        {
            pulsesPerSlope[circle][SLOPE_QUARTER]++;
            xStepsRemaining -= 4;
            yStepsRemaining--;
        }

        // Half slope steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/4.0  &&  xStepsRemaining >= 2)
        {
            pulsesPerSlope[circle][SLOPE_HALF]++;
            xStepsRemaining -= 2;
            yStepsRemaining--;
        }

        // Slope=1 steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/2.0)
        {
            pulsesPerSlope[circle][SLOPE_ONE]++;
            xStepsRemaining--;
            yStepsRemaining--;
        }

        // Slope=2 steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0  &&  yStepsRemaining >= 2)
        {
            pulsesPerSlope[circle][SLOPE_TWO]++;
            xStepsRemaining--;
            yStepsRemaining -= 2;
        }

        // Slope=4 steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 6.0  &&  yStepsRemaining >= 4)
        {
            pulsesPerSlope[circle][SLOPE_FOUR]++;
            xStepsRemaining--;
            yStepsRemaining -= 4;
        }

        // Slope=8 steps
        while(getInstantaneousSlope(radius, xStepsRemaining, yStepsRemaining) < 12.0  &&  yStepsRemaining >= 8)
        {
            pulsesPerSlope[circle][SLOPE_EIGHT]++;
            xStepsRemaining--;
            yStepsRemaining -= 8;
        }

        // Add any leftover x-steps to the beginning of the circle
        pulsesPerSlope[circle][SLOPE_HORIZONTAL] += xStepsRemaining;

        // Remaining y-steps are vertical moves at the end
        pulsesPerSlope[circle][SLOPE_VERTICAL] += yStepsRemaining;
    }
}

// Returns the instantaneous slope of a circle at the current x and y coordinates
// xStepsRemaining and yStepsRemaining must be less than or equal to the radius
float getInstantaneousSlope(int radius, int xStepsRemaining, int yStepsRemaining)
{
    // Return extremely steep slope when yStepsRemaining == 0
    if (yStepsRemaining == 0)
        return 10000.0;
    
    return ((float) radius - xStepsRemaining) / yStepsRemaining;
}
