// Movement function status codes
enum MovementStatus
{
  SUCCESS = 0,
  HIT_POS_X_ENDSTOP = 1,
  HIT_POS_Y_ENDSTOP = 2,
  HIT_NEG_X_ENDSTOP = 3,
  HIT_NEG_Y_ENDSTOP = 4,
  INVALID_ARGS = 5,
  INVALID_ALIGNMENT = 6
};

// Current position tracking scale in eighth steps
// P refers to positive, N refers to negative
// Eighths refers to eight steps
enum EighthStepsScale
{
  P_EIGHTHS_PER_WHOLE_STEP = 8,
  P_EIGHTHS_PER_HALF_STEP = 4,
  P_EIGHTHS_PER_QUARTER_STEP = 2,
  P_EIGHTHS_PER_EIGHTH_STEP = 1,
  N_EIGHTHS_PER_WHOLE_STEP = -8,
  N_EIGHTHS_PER_HALF_STEP = -4,
  N_EIGHTHS_PER_QUARTER_STEP = -2,
  N_EIGHTHS_PER_EIGHTH_STEP = -1
};

// Sets position extremes to be used as alignment codes
enum positionExtremes
{
  ZERO_POSITION = 0,
  MAX_POSITION = 1
};

enum AlignmentSpacing
{
  MAX_X_ALIGNMENT = 15,
  MIN_X_ALIGNMENT = 15,
  MAX_Y_ALIGNMENT = 15,
  MIN_Y_ALIGNMENT = 15
};

// Sets the scale of the motor driver corresponding to "motor"
void setScale(uint8_t motor[], uint8_t scale)
{
  if (scale == WHOLE_STEPS)
  {
    digitalWrite(motor[MS1_PIN], LOW);
    digitalWrite(motor[MS2_PIN], LOW);
    digitalWrite(motor[MS3_PIN], LOW);
  }
  else if (scale == HALF_STEPS)
  {
    digitalWrite(motor[MS1_PIN], HIGH);
    digitalWrite(motor[MS2_PIN], LOW);
    digitalWrite(motor[MS3_PIN], LOW);
  }
  else if (scale == QUARTER_STEPS)
  {
    digitalWrite(motor[MS1_PIN], LOW);
    digitalWrite(motor[MS2_PIN], HIGH);
    digitalWrite(motor[MS3_PIN], LOW);
  }
  else if (scale == EIGHTH_STEPS)
  {
    digitalWrite(motor[MS1_PIN], HIGH);
    digitalWrite(motor[MS2_PIN], HIGH);
    digitalWrite(motor[MS3_PIN], LOW);
  }
  else if (scale == SIXTEENTH_STEPS)
  {
    digitalWrite(motor[MS1_PIN], HIGH);
    digitalWrite(motor[MS2_PIN], HIGH);
    digitalWrite(motor[MS3_PIN], HIGH);
  }
}

void enableMotors()
{
  digitalWrite(X_MOTOR_SLEEP_RESET, HIGH);
  digitalWrite(X_MOTOR_ENABLE, LOW);
  
  digitalWrite(Y_MOTOR_SLEEP_RESET, HIGH);
  digitalWrite(Y_MOTOR_ENABLE, LOW);
}

void disableMotors()
{
  digitalWrite(X_MOTOR_SLEEP_RESET, LOW);
  digitalWrite(X_MOTOR_ENABLE, HIGH);

  digitalWrite(Y_MOTOR_SLEEP_RESET, LOW);
  digitalWrite(Y_MOTOR_ENABLE, HIGH);
}

// Drives the motor corresponding to "motor" to be aligned properly at either the max position or 0
// This function is decomposed into a set of 4 moves:
// ================================================================
// 1. Move to endstop corresponding with alignmentCode until collision
// 2. Move away from that endstop with a certain offset to prepare for the next step
// 3. Move towards that endstop using eighth-steps until collision (provides higher accuracy than step 1)
// 4. Move away from that endstop with an offset corresponding to the nearest valid chessboard corner
void alignAxis(uint8_t motor[], uint8_t alignmentCode)
{
  uint8_t endstopPin;
  uint8_t tempAlignWholeSteps, i;
  uint16_t *currentMotorPosPtr;
    
  // Print debug info about which motor is being aligned to where
  if (DEBUG >= FUNCTION_LEVEL)
  {
    Serial.print("Aligning ");
    Serial.print( (motor == xMotor) ? "x-axis motor" : "y-axis motor");
    Serial.print(" to ");
    Serial.print( (alignmentCode == ZERO_POSITION) ? "zero position" : "max position");
    Serial.println("\n");
  }
  
  // Stores corresponding motor position based off of which motor is being aligned,
  // so that correct position can be incremented by function
  currentMotorPosPtr = (motor == xMotor) ? &currPositionX : &currPositionY;

  // Loop until endstop collision, then fine tune it
  // Use POS_DIR and NEG_DIR to set correct direction of motor alignment despite axis

  // Sets scale and direction for motor and current position
  // Moving motor towards max or 0 for rough estimate
  if (alignmentCode == MAX_POSITION)
  {
    digitalWrite(motor[DIR_PIN], POS_DIR);
    endstopPin = MAX_ENDSTOP_PIN;
  }
  else
  {
    digitalWrite(motor[DIR_PIN], NEG_DIR);
    endstopPin = ZERO_ENDSTOP_PIN;
  }
  setScale(motor, WHOLE_STEPS);

  enableMotors();

  while (digitalRead(motor[endstopPin]) == LOW)
  {
    // Moves motor
    digitalWrite(motor[STEP_PIN], LOW);
    delay(STEP_DELAY);
    digitalWrite(motor[STEP_PIN], HIGH);
  }

  // Flips direction to move motor away from endstop to prepare for fine-tuning
  if (alignmentCode == MAX_POSITION)
    digitalWrite(motor[DIR_PIN], NEG_DIR);
  else
    digitalWrite(motor[DIR_PIN], POS_DIR);

  for (i = 0; i < HOME_CALIBRATION_OFFSET; i++)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(STEP_DELAY);
    digitalWrite(motor[STEP_PIN], HIGH);
  }

  // Slowly moves EM back to nearest endstop for fine-tuning
  if (alignmentCode == MAX_POSITION)
    digitalWrite(motor[DIR_PIN], POS_DIR);
  else
    digitalWrite(motor[DIR_PIN], NEG_DIR);

  setScale(motor, EIGHTH_STEPS);
  
  while (digitalRead(motor[endstopPin]) == LOW)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(STEP_DELAY);
    digitalWrite(motor[STEP_PIN], HIGH);
  }

  // Moves EM back to nearest grid edge after fine-tuned alignment
  if (alignmentCode == MAX_POSITION)
    digitalWrite(motor[DIR_PIN], NEG_DIR);
  else
    digitalWrite(motor[DIR_PIN], POS_DIR);
  
  setScale(motor, WHOLE_STEPS);

  if (motor == xMotor)
    tempAlignWholeSteps = (alignmentCode == MAX_POSITION) ? MAX_X_ALIGNMENT : MIN_X_ALIGNMENT;
  else if (motor == yMotor)
    tempAlignWholeSteps = (alignmentCode == MAX_POSITION) ? MAX_Y_ALIGNMENT : MIN_Y_ALIGNMENT;

  for (i = 0; i < tempAlignWholeSteps; i++)
  {
    digitalWrite(motor[STEP_PIN], LOW);
    delay(STEP_DELAY);
    digitalWrite(motor[STEP_PIN], HIGH);
  }

  // Sets the motor position to either the max position or 0
  *currentMotorPosPtr = (alignmentCode == MAX_POSITION) ? maxPosition : 0;
}

// Aligns both axis to home
void home()
{
  alignAxis(xMotor, ZERO_POSITION);
  alignAxis(yMotor, ZERO_POSITION);
}

// Moves the magnet from the "start" point to the "end" point
// This can only move in straight lines
// A specific motor is passed to this function since we are only moving one here
// Returns a 'MovementStatus' code, '0' if successful and nonzero values for various error codes
// For all status codes, check 'MovementStatus' in chessboard.ino
uint8_t moveStraight(uint8_t motor[], uint8_t endCol, uint8_t endRow)
{
  uint8_t dir, absDelta;
  uint8_t startCol, startRow;
  int8_t eighthStepsPerPulse;
  uint16_t numSteps, i;
  uint16_t *currentMotorPosPtr;

  // Checks if the EM is aligned properly
  if ((currPositionX % stepsPerUnitSpace)  ||  (currPositionY % stepsPerUnitSpace))
    return INVALID_ALIGNMENT;

  // Converts current position to be in terms of unit spaces instead of eighth steps
  startCol = currPositionX / (stepsPerUnitSpace * 8);
  startRow = currPositionY / (stepsPerUnitSpace * 8);

  // If we want to move to where we already are, consider it successful
  if (startCol == endCol  &&  startRow == endRow)
    return SUCCESS;

  // Print debug info about straight movement
  if (DEBUG >= FUNCTION_LEVEL)
  {
    Serial.print("Moving straight from (");
    Serial.print(startCol);
    Serial.print(", ");
    Serial.print(startRow);
    Serial.print(") to (");
    Serial.print(endCol);
    Serial.print(", ");
    Serial.print(endRow);
    Serial.print(") ");
    Serial.print("along the ");
    Serial.println( (motor == xMotor) ? "x-axis" : "y-axis");
  }

  // Same as homeAxis(), sets the loop to only update a single motors position at a time
  // Direction is still determined seperately by if statements
  currentMotorPosPtr = (motor == xMotor) ? &currPositionX : &currPositionY;

  // This could be two cases, x or y movement
  // Abs ensures that numSteps will be positive
  if (endRow == startRow)
  {
    // X movement
    absDelta = abs(endCol - startCol);
    dir = (endCol > startCol) ? POS_DIR : NEG_DIR;
    setScale(xMotor, WHOLE_STEPS);
    // Sets motor and direction if X movement
    eighthStepsPerPulse = (dir == POS_DIR) ? P_EIGHTHS_PER_WHOLE_STEP : N_EIGHTHS_PER_WHOLE_STEP;
  }
  else if (endCol == startCol)
  {
    // Y movement
    absDelta = abs(endRow - startRow);
    dir = (endRow > startRow) ? POS_DIR : NEG_DIR;
    setScale(yMotor, WHOLE_STEPS);
    // Sets motor and direction if Y movement
    eighthStepsPerPulse = (dir == POS_DIR) ? P_EIGHTHS_PER_WHOLE_STEP : N_EIGHTHS_PER_WHOLE_STEP;
  }
  else
  {
    return INVALID_ARGS;
  }

  numSteps = absDelta * stepsPerUnitSpace;

  // Enable motor driver inputs/output
  enableMotors();

  // Set direction of motor
  digitalWrite(motor[DIR_PIN], dir);

  // Rotate motor some number of steps
  for (i = 0; i < numSteps; i++)
  {
    if (digitalRead(X_AXIS_MAX_ENDSTOP) == HIGH)
      return HIT_POS_X_ENDSTOP;

    if (digitalRead(Y_AXIS_MAX_ENDSTOP) == HIGH)
      return HIT_POS_Y_ENDSTOP;
    
    if (digitalRead(X_AXIS_ZERO_ENDSTOP) == HIGH)
      return HIT_NEG_X_ENDSTOP;

    if (digitalRead(Y_AXIS_ZERO_ENDSTOP) == HIGH)
      return HIT_NEG_Y_ENDSTOP;

    digitalWrite(motor[STEP_PIN], LOW);
    delay(STEP_DELAY);
    digitalWrite(motor[STEP_PIN], HIGH);
  }

  // Updates current position of relevant motor
  // Not incremented inside of loop to save runtime and unneeded computation
  // If motor collides with endstop, alignAxis is triggered and fixes motor position
  *currentMotorPosPtr += (eighthStepsPerPulse * numSteps);

  return SUCCESS;
}

// Moves the magnet from the "start" point to the "end" point
// This can move in diagonal lines of slopes: 1, 2, and 1/2
// Returns a 'MovementStatus' code, '0' if successful and nonzero values for various error codes
// For all status codes, check 'MovementStatus' in chessboard.ino
uint8_t moveDiagonal(uint8_t endCol, uint8_t endRow)
{
  uint8_t dirX, dirY, absDeltaX, absDeltaY;
  uint8_t startRow, startCol;
  int8_t eighthStepsPerPulseX, eighthStepsPerPulseY;
  uint16_t numStepsX, numStepsY, i;

  // Checks if the EM is aligned properly
  if ((currPositionX % stepsPerUnitSpace)  ||  (currPositionY % stepsPerUnitSpace))
    return INVALID_ALIGNMENT;

  // Converts current position to be in terms of unit spaces instead of eighth steps
  startCol = currPositionX / (stepsPerUnitSpace * 8);
  startRow = currPositionY / (stepsPerUnitSpace * 8);

  // If we want to move to where we already are, consider it successful
  if (startCol == endCol  &&  startRow == endRow)
    return SUCCESS;

  // Print debug info about diagonal movement
  if (DEBUG >= FUNCTION_LEVEL)
  {
    Serial.print("Moving diagonal from (");
    Serial.print(startCol);
    Serial.print(", ");
    Serial.print(startRow);
    Serial.print(") to (");
    Serial.print(endCol);
    Serial.print(", ");
    Serial.print(endRow);
    Serial.println(")");
  }

  // Sets scale and numEighthSteps for both X and Y
  // Abs ensures that numStepsX and numStepsY will be positive
  // to ensure proper for loop execution
  absDeltaX = abs(endCol - startCol);
  dirX = (endCol > startCol) ? POS_DIR : NEG_DIR;

  absDeltaY = abs(endRow - startRow);
  dirY = (endRow > startRow) ? POS_DIR : NEG_DIR;

  numStepsX = absDeltaX * stepsPerUnitSpace;
  numStepsY = absDeltaY * stepsPerUnitSpace;

  enableMotors();

  digitalWrite(xMotor[DIR_PIN], dirX);
  digitalWrite(yMotor[DIR_PIN], dirY);

  if (numStepsX == numStepsY)
  {
    setScale(xMotor, WHOLE_STEPS);
    setScale(yMotor, WHOLE_STEPS);
    // Sets sign based off of direction of each motor
    eighthStepsPerPulseX = (dirX == POS_DIR) ? P_EIGHTHS_PER_WHOLE_STEP : N_EIGHTHS_PER_WHOLE_STEP;
    eighthStepsPerPulseY = (dirY == POS_DIR) ? P_EIGHTHS_PER_WHOLE_STEP : N_EIGHTHS_PER_WHOLE_STEP;
  }
  else if (numStepsY > numStepsX  &&  (numStepsY / numStepsX) == 2)
  {
    setScale(xMotor, HALF_STEPS);
    setScale(yMotor, WHOLE_STEPS);
    // Sets sign based off of direction of each motor
    eighthStepsPerPulseX = (dirX == POS_DIR) ? P_EIGHTHS_PER_HALF_STEP  : N_EIGHTHS_PER_HALF_STEP;
    eighthStepsPerPulseY = (dirY == POS_DIR) ? P_EIGHTHS_PER_WHOLE_STEP : N_EIGHTHS_PER_WHOLE_STEP;
  }
  else if (numStepsY < numStepsX  &&  (numStepsX / numStepsY) == 2)
  {
    setScale(xMotor, WHOLE_STEPS);
    setScale(yMotor, HALF_STEPS);
    // Sets sign based off of direction of each motor
    eighthStepsPerPulseX = (dirX == POS_DIR) ? P_EIGHTHS_PER_WHOLE_STEP : N_EIGHTHS_PER_WHOLE_STEP;
    eighthStepsPerPulseY = (dirY == POS_DIR) ? P_EIGHTHS_PER_HALF_STEP  : N_EIGHTHS_PER_HALF_STEP;
  }
  else
  {
    return INVALID_ARGS;
  }

  for (i = 0; i < numStepsX; i++)
  {
    if (digitalRead(X_AXIS_MAX_ENDSTOP) == HIGH)
      return HIT_POS_X_ENDSTOP;

    if (digitalRead(Y_AXIS_MAX_ENDSTOP) == HIGH)
      return HIT_POS_Y_ENDSTOP;
    
    if (digitalRead(X_AXIS_ZERO_ENDSTOP) == HIGH)
      return HIT_NEG_X_ENDSTOP;

    if (digitalRead(Y_AXIS_ZERO_ENDSTOP) == HIGH)
      return HIT_NEG_Y_ENDSTOP;

    digitalWrite(xMotor[STEP_PIN], LOW);
    digitalWrite(yMotor[STEP_PIN], LOW);
    delay(STEP_DELAY);
    digitalWrite(xMotor[STEP_PIN], HIGH);
    digitalWrite(yMotor[STEP_PIN], HIGH);
  }

  // Updates current position for both X and Y motors
  // Not incremented inside of loop to save runtime and unneeded computation
  // If motor collides with endstop, alignAxis is triggered and fixes motor position
  currPositionX += (eighthStepsPerPulseX * numStepsX);
  currPositionY += (eighthStepsPerPulseY * numStepsX);

  return SUCCESS;
}
