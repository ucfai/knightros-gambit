// Slopes: 0, 1/8, 1/4, 1/2, 1, 2, 4, 8, Vertical
int stepsPerSlope[NUM_CIRCLES][NUM_SLOPES_PER_QUARTER_CIRCLE];

void makeCircle(int circle, int firstQuarter, int lastQuarter)
{
    bool firstPass = true;

    setScale(xMotor, EIGHTH_STEPS);
    digitalWrite(xMotor[STEP_PIN], LOW);

    for (int quarterCircle = firstQuarter; quarterCircle != firstQuarter || firstPass; quarterCircle = (quarterCircle + 1) % 4)
    {
        firstPass = false;

        // Set Y-direction
        if (quarterCircle == 0 || quarterCircle == 2)
            digitalWrite(yMotor[DIR_PIN], DOWN);
        else
            digitalWrite(yMotor[DIR_PIN], UP);

        // Set X-direction
        if (quarterCircle == 0 || quarterCircle == 3)
            digitalWrite(xMotor[DIR_PIN], LEFT);
        else
            digitalWrite(xMotor[DIR_PIN], RIGHT);

        int slope;
        if (quarterCircle == 0 || quarterCircle == 2)
            slope = 0;
        else
            slope = NUM_SLOPES_PER_QUARTER_CIRCLE - 1;

        while (slope >= 0 && slope < NUM_SLOPES_PER_QUARTER_CIRCLE)
        {
            // Y-scale
            if (slope == 5)
                setScale(yMotor, QUARTER_STEPS);    
            else if (slope == 6)
                setScale(yMotor, HALF_STEPS);    
            else if (slope == 7)
                setScale(yMotor, WHOLE_STEPS);    
            else
                setScale(yMotor, EIGHTH_STEPS);    

            // X-scale
            if (slope == 1)
                setScale(xMotor, WHOLE_STEPS);
            else if (slope == 2)
                setScale(xMotor, HALF_STEPS);
            else if (slope == 3)
                setScale(xMotor, QUARTER_STEPS);
            else 
                setScale(xMotor, EIGHTH_STEPS);

            for (int i = 0; i < stepsPerSlope[circle][slope]; i++)
            {
                // X-Step
                if (slope != 8)
                {
                    digitalWrite(xMotor[STEP_PIN], LOW);
                    digitalWrite(xMotor[STEP_PIN], HIGH);
                }

                // Y-Step
                if (slope != 0)
                {
                    digitalWrite(yMotor[STEP_PIN], LOW);
                    digitalWrite(yMotor[STEP_PIN], HIGH);
                }
            }

            // Increment/Decrement slope
            if (quarterCircle == 0 || quarterCircle == 2)
                slope++;
            else
                slope--;
        }
    }
}

void calculateStepsPerSlope(){

    int outerRadius, deltaR;

    // Calculate the radius of the largest circle in eighth steps
    outerRadius = MILLIMETERS_PER_SQUARE * STEPS_PER_MILLIMETER * 2;

    // Ensure that each radius is evenly spaced from the center
    outerRadius = outerRadius - outerRadius % NUM_CIRCLES;

    // Calculate the spacing between the circles
    deltaR = outerRadius / NUM_CIRCLES;

    // Loop over each circle
    for (int circle = 0; circle < NUM_CIRCLES; circle++)
    {
        int radius, xStepsRemaining, yStepsRemaining;

        radius = outerRadius - deltaR * circle;
        xStepsRemaining = radius;
        yStepsRemaining = radius;

        // Horizontal steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 1.0/16.0)
        {
           stepsPerSlope[circle][0]++;
           xStepsRemaining--;
        }

        // Eighth slope steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/16.0 && xStepsRemaining >= 8)
        {
            stepsPerSlope[circle][1]++;
            xStepsRemaining -= 8;
            yStepsRemaining--;
        }

        // Quarter slope steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/8.0 && xStepsRemaining >= 4)
        {
            stepsPerSlope[circle][2]++;
            xStepsRemaining -= 4;
            yStepsRemaining--;
        }

        // Half slope steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/4.0 && xStepsRemaining >= 2)
        {
            stepsPerSlope[circle][3]++;
            xStepsRemaining -= 2;
            yStepsRemaining--;
        }

        // Slope=1 steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0/2.0)
        {
            stepsPerSlope[circle][4]++;
            xStepsRemaining--;
            yStepsRemaining--;
        }

        // Slope=2 steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 3.0 && yStepsRemaining >= 2)
        {
            stepsPerSlope[circle][5]++;
            xStepsRemaining--;
            yStepsRemaining -= 2;
        }

        // Slope=4 steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 6.0 && yStepsRemaining >= 4)
        {
            stepsPerSlope[circle][6]++;
            xStepsRemaining--;
            yStepsRemaining -= 4;
        }

        // Slope=8 steps
        while(getSlope(radius, xStepsRemaining, yStepsRemaining) < 12.0 && yStepsRemaining >= 8)
        {
            stepsPerSlope[circle][7]++;
            xStepsRemaining--;
            yStepsRemaining -= 8;
        }

        // Add any leftover x-steps to the beginning of the circle
        stepsPerSlope[circle][0] += xStepsRemaining;

        // Remaining y-steps are vertical moves at the end
        stepsPerSlope[circle][8] += yStepsRemaining;
    }
}

float getSlope(int radius, int xStepsRemaining, int yStepsRemaining)
{
    // Return extremely steep slope when yStepsRemaining == 0
    if (yStepsRemaining == 0)
        return 10000.0;
    
    return ((float) radius - xStepsRemaining) / yStepsRemaining;
}
