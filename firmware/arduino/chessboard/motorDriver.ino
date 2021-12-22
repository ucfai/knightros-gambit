/*
  Motor array format in order of indicies:
  Index:
  0         1        2        3
  Step pin, Dir pin, MS1 pin, MS2 pin
*/

void moveStraight(int motor[], int spaces, int dir)
{
  // How many steps per space
  int numSteps = spaces * stepsPerSpace;
  int i;

  // Enable motor driver inputs/output
  digitalWrite(MOTOR_SLEEP, HIGH);
  digitalWrite(MOTOR_RESET, HIGH);
  digitalWrite(MOTOR_ENABLE, LOW);

  // Set direction of motor
  digitalWrite(motor[1], dir);

  // Rotate motor some number of steps
  for (i = 0; i < numSteps; i++) 
  {
    digitalWrite(motor[0], LOW);
    delay(1);  // 1 milliSecond
    digitalWrite(motor[0], HIGH);
  }
}

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

void moveDiagonal(int dirX, int dirY, int spacesX, int spacesY)
{
  int i;

  int numStepsX = spacesX * stepsPerSpace;
  int numStepsY = spacesY * stepsPerSpace;

  digitalWrite(MOTOR_SLEEP, HIGH);
  digitalWrite(MOTOR_RESET, HIGH);
  digitalWrite(MOTOR_ENABLE, LOW);

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
      digitalWrite(xMotor[0], LOW);
      digitalWrite(yMotor[0], LOW);
      delay(1);
      digitalWrite(xMotor[0], HIGH);
      digitalWrite(yMotor[0], HIGH);
    }
}

void disableMotors() 
{
  digitalWrite(MOTOR_SLEEP, LOW);
  digitalWrite(MOTOR_RESET, LOW);
  digitalWrite(MOTOR_ENABLE, HIGH);
}
