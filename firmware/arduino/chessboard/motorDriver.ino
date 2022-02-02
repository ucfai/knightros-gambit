// Movement function status codes
enum MovementStatus
{
  SUCCESS = 0,
  HIT_X_ENDSTOP = 1,
  HIT_Y_ENDSTOP = 2,
  INVALID_ARGS = 3
};

// Sets the scale of the motor driver corresponding to "motor"
void setScale(int motor[], int scale) 
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
void homeAxis(int motor[])
{
  int * currentMotorPos;
  int currentMotorDir, currentMotorNumEighthSteps;
  int i;

  currentMotorPos = (motor == xMotor) ? &currentX : &currentY;

  // Loop until endstop collision, then fine tune it
  // LEFT and DOWN have same value, as do RIGHT and UP. Use LEFT
  // and RIGHT arbitrarily while tuning end stop for both x and y axes.

  // Sets scale and direction for motor and current position
  digitalWrite(motor[DIR_PIN], LEFT);
  setScale(motor, WHOLE_STEPS);
  currentMotorNumEighthSteps = 8;
  currentMotorNumEighthSteps *= -1;
  while (digitalRead(motor[ENDSTOP_PIN]) == LOW)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(1);
    digitalWrite(motor[STEP_PIN], HIGH);

    // Motor is moved above, current position for whatever motor updated here
    *currentMotorPos += currentMotorNumEighthSteps;
  }

  digitalWrite(motor[DIR_PIN], RIGHT);
  currentMotorNumEighthSteps *= -1;
  for (i = 0; i < HOME_CALIBRATION_OFFSET; i++)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(1);
    digitalWrite(motor[STEP_PIN], HIGH);
   *currentMotorPos += currentMotorNumEighthSteps;
  }

  digitalWrite(motor[DIR_PIN], LEFT);
  setScale(motor, EIGHTH_STEPS);
  currentMotorNumEighthSteps = 1;
  currentMotorNumEighthSteps *= -1;
  while (digitalRead(motor[ENDSTOP_PIN]) == LOW)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(1);
    digitalWrite(motor[STEP_PIN], HIGH);
    *currentMotorPos += currentMotorNumEighthSteps;
  }

  // Set the motor position to "home" position
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
uint8_t moveStraight(int motor[], int startCol, int startRow, int endCol, int endRow)
{
  // How many steps per space
  int dir, numSteps, unitSpaces;
  int i;
  int numEighthStepsX, numEighthStepsY;

  // This could be two cases, x or y movement
  // Abs ensures that numSteps will be positive
  if (endRow == startRow)
  {
    // X movement
    unitSpaces = abs(endCol - startCol);
    dir = (endCol > startCol) ? RIGHT : LEFT;
    setScale(xMotor, WHOLE_STEPS);
  }
  else if (endCol == startCol)
  {
    // Y movement
    unitSpaces = abs(endRow - startRow);
    dir = (endRow > startRow) ? UP : DOWN;
    setScale(yMotor, WHOLE_STEPS);
  }
  else
  {
    return INVALID_ARGS;
  }

  numSteps = unitSpaces * stepsPerUnitSpace;

  // Setting munEightSteps moved at a time to 8 because function only moves in whole steps
  // Sign is set based off of direction of each motor
  numEighthStepsX = 8;
  numEighthStepsY = 8;
  numEighthStepsX *= (dir == RIGHT) ? 1 : -1;
  numEighthStepsY *= (dir == UP) ? 1 : -1;

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
    delay(1);  // 1 milliSecond
    digitalWrite(motor[STEP_PIN], HIGH);

    // Updating current position per step for accuracy
    currentX += numEighthStepsX;
    currentY += numEighthStepsY;
  }

  return SUCCESS;
}

// Moves the magnet from the "start" point to the "end" point
// This can move in diagonal lines of slopes: 1, 2, and 1/2
// Returns a 'MovementStatus' code, '0' if successful, varying nonzero values for various error codes
// For all status codes, check 'MovementStatus' in chessboard.ino
uint8_t moveDiagonal(int startCol, int startRow, int endCol, int endRow)
{
  int unitSpacesX, unitSpacesY;
  int dirX, dirY;
  int numStepsX, numStepsY;
  int numEighthStepsX, numEighthStepsY;
  int i;

  // In following initializations and if statements, numEigthSteps scale is set
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
    numEighthStepsX = 8;
    numEighthStepsY = 8;
  }
  else if (numStepsY > numStepsX && (numStepsY / numStepsX) == 2)
  {
    setScale(xMotor, HALF_STEPS);
    setScale(yMotor, WHOLE_STEPS);
    numEighthStepsX = 4;
    numEighthStepsY = 8;
  }
  else if (numStepsY < numStepsX && (numStepsX / numStepsY) == 2)
  {
    setScale(xMotor, WHOLE_STEPS);
    setScale(yMotor, HALF_STEPS);
    numEighthStepsX = 8;
    numEighthStepsY = 4;
  }
  else
  {
    return INVALID_ARGS;
  }

  // Sign is set based off of direction of each motor
  numEighthStepsX *= (dirX == RIGHT) ? 1 : -1;
  numEighthStepsY *= (dirY == UP) ? 1 : -1;

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

    // Updating current position per step for accuracy
    currentX += numEighthStepsX;
    currentY += numEighthStepsY;
  }

  return SUCCESS;
}