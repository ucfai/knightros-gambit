// Movement function status codes
enum MovementStatus
{
  SUCCESS = 0,
  HIT_X_ENDSTOP = 1,
  HIT_Y_ENDSTOP = 2,
  INVALID_ARGS = 3
};

// Current position tracking scale in eighth steps
enum EighthStepsScale
{
  POS_EIGHTH_STEPS_PER_WHOLE_STEP = 8,
  POS_EIGHTH_STEPS_PER_HALF_STEP = 4,
  POS_EIGHTH_STEPS_PER_QUARTER_STEP = 2,
  POS_EIGHTH_STEPS_PER_EIGHTH_STEP = 1,
  NEG_EIGHTH_STEPS_PER_WHOLE_STEP = -8,
  NEG_EIGHTH_STEPS_PER_HALF_STEP = -4,
  NEG_EIGHTH_STEPS_PER_QUARTER_STEP = -2,
  NEG_EIGHTH_STEPS_PER_EIGHTH_STEP = -1
};

// Sets the scale of the motor driver corresponding to "motor"
void setScale(uint8_t motor[], int scale)
{
  if (scale == WHOLE_STEPS)
  {
    digitalWrite(motor[MS1_PIN], LOW);
    digitalWrite(motor[MS2_PIN], LOW);
  }
  else if (scale == HALF_STEPS)
  {
    digitalWrite(motor[MS1_PIN], HIGH);
    digitalWrite(motor[MS2_PIN], LOW);
  }
  else if (scale == QUARTER_STEPS)
  {
    digitalWrite(motor[MS1_PIN], LOW);
    digitalWrite(motor[MS2_PIN], HIGH);
  }
  else if (scale == EIGHTH_STEPS)
  {
    digitalWrite(motor[MS1_PIN], HIGH);
    digitalWrite(motor[MS2_PIN], HIGH);
  }
}

void enableMotors()
{
  digitalWrite(MOTOR_SLEEP, HIGH);
  digitalWrite(MOTOR_RESET, HIGH);
  digitalWrite(MOTOR_ENABLE, LOW);
}

void disableMotors()
{
  digitalWrite(MOTOR_SLEEP, LOW);
  digitalWrite(MOTOR_RESET, LOW);
  digitalWrite(MOTOR_ENABLE, HIGH);
}

// Drives the motor corresponding to "motor" to it's home position (position 0)
void homeAxis(uint8_t motor[])
{
  int *currentMotorPos;
  int eighthStepsPerPulse;
  int i;

  // Stores corresponding motor position based off of which motor is being homed,
  // so that correct position can be incremented by function
  currentMotorPos = (motor == xMotor) ? &currentX : &currentY;

  // Loop until endstop collision, then fine tune it
  // LEFT and DOWN have same value, as do RIGHT and UP. Use LEFT
  // and RIGHT arbitrarily while tuning end stop for both x and y axes.

  // Sets scale and direction for motor and current position
  // Moving motor towards home for rough estimate
  digitalWrite(motor[DIR_PIN], LEFT);
  setScale(motor, WHOLE_STEPS);
  eighthStepsPerPulse = NEG_EIGHTH_STEPS_PER_WHOLE_STEP;
  while (digitalRead(motor[ENDSTOP_PIN]) == LOW)
  {
    // Moves motor
    digitalWrite(motor[STEP_PIN], LOW);
    delay(1);
    digitalWrite(motor[STEP_PIN], HIGH);

    // Updates current motor position
    *currentMotorPos += eighthStepsPerPulse;
  }

  // Flips direction again to move motor away from home to prepare for fine-tuning
  digitalWrite(motor[DIR_PIN], RIGHT);
  eighthStepsPerPulse = POS_EIGHTH_STEPS_PER_WHOLE_STEP;
  for (i = 0; i < HOME_CALIBRATION_OFFSET; i++)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(1);
    digitalWrite(motor[STEP_PIN], HIGH);
    *currentMotorPos += eighthStepsPerPulse;
  }

  // Moves motor towards home for fine-tuned home position
  digitalWrite(motor[DIR_PIN], LEFT);
  setScale(motor, EIGHTH_STEPS);
  eighthStepsPerPulse = NEG_EIGHTH_STEPS_PER_EIGHTH_STEP;
  while (digitalRead(motor[ENDSTOP_PIN]) == LOW)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(1);
    digitalWrite(motor[STEP_PIN], HIGH);
    *currentMotorPos += eighthStepsPerPulse;
  }

  // Sets the motor position to "home" position
  *currentMotorPos = 0;
}

// Homes both axis
void home()
{
  homeAxis(xMotor);
  homeAxis(yMotor);
}

// Moves the magnet from the "start" point to the "end" point
// This can only move in straight lines
// A specific motor is passed to this function since we are only moving one here
// Returns a 'MovementStatus' code, '0' if successful, varying nonzero values for various error codes
// For all status codes, check 'MovementStatus' in chessboard.ino
uint8_t moveStraight(uint8_t motor[], int endCol, int endRow)
{
  // How many steps per space
  int dir, numSteps, unitSpaces;
  int i;
  int eighthStepsPerPulse;
  int *currentMotorPos;
  float startCol, startRow;

  // Converts current position to be in terms of unit spaces instead of eighth steps
  startCol = currentX / (stepsPerUnitSpace * 8);
  startRow = currentY / (stepsPerUnitSpace * 8);

  // Same as homeAxis(), sets the loop to only update a single motors position at a time
  // Direction is still determined seperately by if statements
  currentMotorPos = (motor == xMotor) ? &currentX : &currentY;

  // This could be two cases, x or y movement
  // Abs ensures that numSteps will be positive
  if (endRow == startRow)
  {
    // X movement
    unitSpaces = abs(endCol - startCol);
    dir = (endCol > startCol) ? RIGHT : LEFT;
    setScale(xMotor, WHOLE_STEPS);
    // Sets motor and direction if X movement
    eighthStepsPerPulse = (dir == RIGHT) ? POS_EIGHTH_STEPS_PER_WHOLE_STEP : NEG_EIGHTH_STEPS_PER_WHOLE_STEP;
  }
  else if (endCol == startCol)
  {
    // Y movement
    unitSpaces = abs(endRow - startRow);
    dir = (endRow > startRow) ? UP : DOWN;
    setScale(yMotor, WHOLE_STEPS);
    // Sets motor and direction if Y movement
    eighthStepsPerPulse = (dir == UP) ? POS_EIGHTH_STEPS_PER_WHOLE_STEP : NEG_EIGHTH_STEPS_PER_WHOLE_STEP;
  }
  else
  {
    return INVALID_ARGS;
  }

  numSteps = unitSpaces * stepsPerUnitSpace;

  // Enable motor driver inputs/output
  enableMotors();

  // Set direction of motor
  digitalWrite(motor[DIR_PIN], dir);

  // Rotate motor some number of steps
  for (i = 0; i < numSteps; i++)
  {
    if (digitalRead(X_AXIS_ENDSTOP_SWITCH) == HIGH)
      return HIT_X_ENDSTOP;

    if (digitalRead(Y_AXIS_ENDSTOP_SWITCH) == HIGH)
      return HIT_Y_ENDSTOP;

    digitalWrite(motor[STEP_PIN], LOW);
    delay(1); // 1 milliSecond
    digitalWrite(motor[STEP_PIN], HIGH);

    // Updating current position per step
    *currentMotorPos += eighthStepsPerPulse;
  }

  return SUCCESS;
}

// Moves the magnet from the "start" point to the "end" point
// This can move in diagonal lines of slopes: 1, 2, and 1/2
// Returns a 'MovementStatus' code, '0' if successful, varying nonzero values for various error codes
// For all status codes, check 'MovementStatus' in chessboard.ino
uint8_t moveDiagonal(int endCol, int endRow)
{
  int unitSpacesX, unitSpacesY;
  int dirX, dirY;
  int numStepsX, numStepsY;
  int eighthStepsPerPulseX, eighthStepsPerPulseY;
  int i;
  float startRow, startCol;

  // Converts current position to be in terms of unit spaces instead of eighth steps
  startCol = currentX / (stepsPerUnitSpace * 8);
  startRow = currentY / (stepsPerUnitSpace * 8);

  // Sets scale and numEighthSteps for both X and Y
  // Abs ensures that numStepsX and numStepsY will be positive
  // to ensure proper for loop execution
  unitSpacesX = abs(endCol - startCol);
  dirX = (endCol > startCol) ? RIGHT : LEFT;

  unitSpacesY = abs(endRow - startRow);
  dirY = (endRow > startRow) ? UP : DOWN;

  numStepsX = unitSpacesX * stepsPerUnitSpace;
  numStepsY = unitSpacesY * stepsPerUnitSpace;

  enableMotors();

  digitalWrite(xMotor[DIR_PIN], dirX);
  digitalWrite(yMotor[DIR_PIN], dirY);

  if (numStepsX == numStepsY)
  {
    setScale(xMotor, WHOLE_STEPS);
    setScale(yMotor, WHOLE_STEPS);
    // Sets sign based off of direction of each motor
    eighthStepsPerPulseX = (dirX == RIGHT) ? POS_EIGHTH_STEPS_PER_WHOLE_STEP : NEG_EIGHTH_STEPS_PER_WHOLE_STEP;
    eighthStepsPerPulseY = (dirY == UP) ? POS_EIGHTH_STEPS_PER_WHOLE_STEP : NEG_EIGHTH_STEPS_PER_WHOLE_STEP;
  }
  else if (numStepsY > numStepsX && (numStepsY / numStepsX) == 2)
  {
    setScale(xMotor, HALF_STEPS);
    setScale(yMotor, WHOLE_STEPS);
    // Sets sign based off of direction of each motor
    eighthStepsPerPulseX = (dirX == RIGHT) ? POS_EIGHTH_STEPS_PER_HALF_STEP : NEG_EIGHTH_STEPS_PER_HALF_STEP;
    eighthStepsPerPulseY = (dirY == UP) ? POS_EIGHTH_STEPS_PER_WHOLE_STEP : NEG_EIGHTH_STEPS_PER_WHOLE_STEP;
  }
  else if (numStepsY < numStepsX && (numStepsX / numStepsY) == 2)
  {
    setScale(xMotor, WHOLE_STEPS);
    setScale(yMotor, HALF_STEPS);
    // Sets sign based off of direction of each motor
    eighthStepsPerPulseX = (dirX == RIGHT) ? POS_EIGHTH_STEPS_PER_WHOLE_STEP : NEG_EIGHTH_STEPS_PER_WHOLE_STEP;
    eighthStepsPerPulseY = (dirY == UP) ? POS_EIGHTH_STEPS_PER_HALF_STEP : NEG_EIGHTH_STEPS_PER_HALF_STEP;
  }
  else
  {
    return INVALID_ARGS;
  }

  for (i = 0; i < numStepsX; i++)
  {
    if (digitalRead(X_AXIS_ENDSTOP_SWITCH) == HIGH)
      return HIT_X_ENDSTOP;

    if (digitalRead(Y_AXIS_ENDSTOP_SWITCH) == HIGH)
      return HIT_Y_ENDSTOP;

    digitalWrite(xMotor[STEP_PIN], LOW);
    digitalWrite(yMotor[STEP_PIN], LOW);
    delay(1);
    digitalWrite(xMotor[STEP_PIN], HIGH);
    digitalWrite(yMotor[STEP_PIN], HIGH);

    // Updates current position for both x and y
    currentX += eighthStepsPerPulseX;
    currentY += eighthStepsPerPulseY;
  }

  return SUCCESS;
}