// Slopes: 0, 1/8, 1/4, 1/2, 1, 2, 4, 8, Vertical
int stepsPerSlope[NUM_CIRCLES][NUM_SLOPES_PER_QUARTER_CIRCLE];

void makeCircle(int circle)
{
    setScale(xMotor, EIGHTH_STEPS);
    digitalWrite(xMotor[STEP_PIN], LOW);

    for (int quarterCircle = 0; quarterCircle < 4; quarterCircle++)
    {
        // Set Y-direction
        if (quarterCircle == 0 || quarterCircle == 2)
        {
            digitalWrite(yMotor[DIR_PIN], DOWN);
        }
        else
        {
            digitalWrite(yMotor[DIR_PIN], UP);
        }

        // Set X-direction
        if (quarterCircle == 0 || quarterCircle == 3)
        {
            digitalWrite(xMotor[DIR_PIN], LEFT);
        }
        else
        {
            digitalWrite(xMotor[DIR_PIN], RIGHT);
        }

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
