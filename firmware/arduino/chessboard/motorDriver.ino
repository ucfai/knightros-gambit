/*
  Motor array format in order of indicies:
  Index:
  0         1        2        3
  Step pin, Dir pin, MS1 pin, MS2 pin
*/
void setScale(int motor[], int scale) 
{
  if (scale == 1)  // whole steps
  {
    digitalWrite(motor[2], LOW);
    digitalWrite(motor[3], LOW);
  }
  else if (scale == 2)  // 1/2 steps
  {
    digitalWrite(motor[2], HIGH);
    digitalWrite(motor[3], LOW);
  }
  else if (scale == 4)  // 1/4 steps
  {
    digitalWrite(motor[2], LOW);
    digitalWrite(motor[3], HIGH);
  }
  else if (scale == 8)  // 1/8 steps
  {
    digitalWrite(motor[2], HIGH);
    digitalWrite(motor[3], HIGH);
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

void homeAxis(int motor[])
{
  int i;

  // Loop until endstop collision, then fine tune it
  // Since we're reusing this code for the x and y axis, 
  // LEFT and RIGHT don't always apply to the motor passed in, but the values are the same 

  digitalWrite(motor[1], LEFT);
  setScale(motor, 1);
  while (digitalRead(motor[4]) == LOW)
  {
    digitalWrite(motor[0], LOW);
    delay(1);  // 1 milliSecond
    digitalWrite(motor[0], HIGH);
  }

  digitalWrite(motor[1], RIGHT);
  for (i = 0; i < HOME_OFFSET; i++)
  {
    digitalWrite(motor[0], LOW);
    delay(1);  // 1 milliSecond
    digitalWrite(motor[0], HIGH);
  }

  digitalWrite(motor[1], LEFT);
  setScale(motor, 8);
  while (digitalRead(motor[4]) == LOW)
  {
    digitalWrite(motor[0], LOW);
    delay(1);  // 1 milliSecond
    digitalWrite(motor[0], HIGH);
  }
}

void home()
{
  homeAxis(xMotor);
  homeAxis(yMotor);
}

/*
  Motor array format in order of indicies:
  Index:
  0         1        2        3
  Step pin, Dir pin, MS1 pin, MS2 pin
*/

void moveStraight(int motor[], int startCol, int startRow, int endCol, int endRow)
{
  // A specific motor is passed to this function since we are only moving one here
  
  // How many steps per space
  int spaces, dir, numSteps;
  int i;

  // This could be two cases, x or y movement
  if (endRow == startRow)
  {
    // X movement
    spaces = abs(endCol - startCol);
    dir = (endCol > startCol) ? RIGHT : LEFT;
    setScale(xMotor, 1);
  }
  else if (endCol == startCol)
  {
    // Y movement
    spaces = abs(endRow - startRow);
    dir = (endRow > startRow) ? UP : DOWN;
    setScale(yMotor, 1);
  }

  numSteps = spaces * stepsPerSpace;

  // Enable motor driver inputs/output
  enableMotors();

  // Set direction of motor
  digitalWrite(motor[1], dir);

  // Rotate motor some number of steps
  for (i = 0; i < numSteps; i++) 
  {
    if (digitalRead(X_AXIS_ENDSTOP_SWITCH) == HIGH  ||  digitalRead(Y_AXIS_ENDSTOP_SWITCH))
      break;
    
    digitalWrite(motor[0], LOW);
    delay(1);  // 1 milliSecond
    digitalWrite(motor[0], HIGH);
  }
}

void moveDiagonal(int startCol, int startRow, int endCol, int endRow)
{
  int dirX, dirY, spacesX, spacesY;
  int numStepsX, numStepsY;
  int i;

  // Abs ensures that numStepsx and numStepsY will be positive
  spacesX = abs(endCol - startCol);
  dirX = (endCol > startCol) ? RIGHT : LEFT;

  spacesY = abs(endRow - startRow);
  dirY = (endRow > startRow) ? UP : DOWN;

  numStepsX = spacesX * stepsPerSpace;
  numStepsY = spacesY * stepsPerSpace;

  enableMotors();

  digitalWrite(xMotor[1], dirX);
  digitalWrite(yMotor[1], dirY);
  
  if (numStepsX == numStepsY)
  {
    setScale(xMotor, 1);
    setScale(yMotor, 1);
  }
  else if (numStepsY > numStepsX && (numStepsY / numStepsX) == 2)
  {
    setScale(xMotor, 2);
    setScale(yMotor, 1);
  }
  else if (numStepsY < numStepsX && (numStepsX / numStepsY) == 2)
  {
    setScale(xMotor, 1);
    setScale(yMotor, 2);
  }

  for(i = 0; i < numStepsX; i++)
  {
    if (digitalRead(X_AXIS_ENDSTOP_SWITCH) == HIGH  ||  digitalRead(Y_AXIS_ENDSTOP_SWITCH))
      break;

    digitalWrite(xMotor[0], LOW);
    digitalWrite(yMotor[0], LOW);
    delay(1);
    digitalWrite(xMotor[0], HIGH);
    digitalWrite(yMotor[0], HIGH);
  }
}
