bool moveDirect(int startCol, int startRow, int endCol, int endRow)
{
  uint8_t statusCodeResult;
  uint8_t currCol, currRow;
  uint8_t *motorPtr;
  int8_t absDiagSpaces, diagSpacesX, diagSpacesY;
  int8_t deltaX, deltaY, absDeltaX, absDeltaY;
  bool isSuccessful;

  // Find the signed difference between the final and initial points
  deltaX = endCol - startCol;
  deltaY = endRow - startRow;

  // We need the magnitude of deltaX and deltaY for handling invalid slopes for moveDiagonal  
  absDeltaX = abs(deltaX);
  absDeltaY = abs(deltaY);

  // If target point is equal to the start point
  if (deltaX == 0  &&  deltaY == 0)
    return true;

  // Checks if the EM is aligned properly
  if ((currPositionX % stepsPerUnitSpace) || (currPositionY % stepsPerUnitSpace))
    return false;

  currCol = currPositionX / (stepsPerUnitSpace * 8);
  currRow = currPositionY / (stepsPerUnitSpace * 8);

  // Make initial move to first position and move directly since it's faster.
  // This is calling the function recursively in the following way:
  // ==============================================================
  // 1. When the initial function call is made, we end up entering the first recursive call 
  //    to move to the start point.
  // 2. If the target point is the start point, just return true and continue. Otherwise calculate 
  //    the numbers to move to the first point and enter the 2nd recursive call.
  //    (Note that the start point in this call is the current point from the previous call)
  // 3. In this call, we have the start point and end point equal to the current point, so it 
  //    returns true before anything is done.
  // 4. Now we end up in the first recursive call and simply move to the start point.
  // 5. After being moved to the start point, we perform the move from the initial call.
  if (!moveDirect(currCol, currRow, startCol, startRow))
    return false;

  // Enable electromagnet
  ledcWrite(EM_PWM_CHANNEL, PWM_HALF);

  // This variable allows us to store the return value of the function
  // so we can turn off the electromagnet before exiting.
  // Assume no errors, unless proven otherwise
  isSuccessful = true;

  if (deltaX == 0  ||  deltaY == 0)
  {
    // Assign the correct motor based on which motor has movement
    // Note: Because of the first base case, either deltaX or deltaY must be 0, but not both
    motorPtr = (deltaX == 0) ? yMotor : xMotor;

    statusCodeResult = moveStraight(motorPtr, endCol, endRow);
    if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
      isSuccessful = false;
  }
  else
  {
    // Note: moveDiagonal handles slopes of 1, 1/2, and 2
    statusCodeResult = moveDiagonal(endCol, endRow);

    if (statusCodeResult != SUCCESS)
    {  
      // If we have INVALID_ARGS, the slope is not valid
      // To account for this, we'll decompose the move into two moves:
      // 1. Move using a slope of 1 until aligned with the target point
      // 2. Move straight to the target point for the rest of the way
      if (statusCodeResult == INVALID_ARGS)
      {
        // All moveDirect calls that move pieces have valid slopes, so we can only encounter this
        // if we are moving to the start point, which does not move pieces
        digitalWrite(ELECTROMAGNET, LOW);

        // Reset isSuccessful to true since we have calls that can be successful after this
        isSuccessful = true;

        // We need to find the smaller distance for the diagonal movement using the absolute value, 
        // because if either deltaX or deltaY are negative it will yield an incorrect minimum
        // Calculate the X and Y components separately, since they can be different signs  
        absDiagSpaces = min(absDeltaX, absDeltaY);
        diagSpacesX = (deltaX > 0) ? absDiagSpaces : -absDiagSpaces;
        diagSpacesY = (deltaY > 0) ? absDiagSpaces : -absDiagSpaces;

        statusCodeResult = moveDiagonal(startRow + diagSpacesY, startCol + diagSpacesX);
        if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
          isSuccessful = false;
        
        if (isSuccessful == true)
        {
          // Assign the correct motor based on which motor still has movement left
          motorPtr = (absDiagSpaces == absDeltaX) ? yMotor : xMotor;

          statusCodeResult = moveStraight(yMotor, endCol, endRow);
          if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
            isSuccessful = false;
        }
      }
      else
      {
        isSuccessful = statusCodeHandler(statusCodeResult);
      }
    }
  }

  // Turn electromagnet off
  digitalWrite(ELECTROMAGNET, LOW);

  return isSuccessful;
}

// This type of move is typically decomposed into a set of 4 moves:
// ================================================================
// 1. Move to edge from center of square
// 2. Move along X axis (optional)
// 3. Move along Y axis (optional)
// 4. Move to center of square from edge
bool moveAlongEdges(int startCol, int startRow, int endCol, int endRow)
{
  // There are 5 possible points total, where the first is always the passed start point and 
  // the rest can be made from the list above.
  uint8_t rows[5], cols[5];
  int8_t diagDirX, diagDirY;
  int8_t deltaX, deltaY, subDeltaX, subDeltaY, absDeltaX, absDeltaY;
  uint8_t pointCounter, statusCodeResult;
  uint8_t currCol, currRow;
  uint8_t numPoints = 0;

  // Find the signed difference between the final and initial points
  deltaX = endCol - startCol;
  deltaY = endRow - startRow;

  // We need the magnitude of deltaX and deltaY for edge case checks, so we'll store them for later 
  absDeltaX = abs(deltaX);
  absDeltaY = abs(deltaY);

  // Calculate the sub-distances for X and Y movement along chess square edges
  // They can be overridden as necessary
  subDeltaX = deltaX - (2 * diagDirX);
  subDeltaY = deltaY - (2 * diagDirY);

  // Calculate the direction of the diagonals
  diagDirX = (deltaX > 0) ? 1 : -1;
  diagDirY = (deltaY > 0) ? 1 : -1;

  // If target point is equal to the start point
  if (deltaX == 0  &&  deltaY == 0)
    return true;
    
  // Checks if the EM is aligned properly
  if ((currPositionX % stepsPerUnitSpace) || (currPositionY % stepsPerUnitSpace))
    return false;

  currCol = currPositionX / (stepsPerUnitSpace * 8);
  currRow = currPositionY / (stepsPerUnitSpace * 8);

  // Make initial move to first position. Move diagonally since it's faster.
  if (!moveDirect(currCol, currRow, startCol, startRow))
    return false;

  // Calculate a list of up to 4 new points (including the start point) 
  // We use previous points to calculate subsequent points, since they're on the same path

  // Case where we should call moveDirect, since pathing can be simplified to that
  if (absDeltaX <= 2  &&  absDeltaY <= 2)
    return moveDirect(startCol, startRow, endCol, endRow);

  // Initial position
  cols[0] = startCol;
  rows[0] = startRow;

  // Because of the checks above, we know absDeltaX and absDeltaY can't both be 0 at the same time
  // Case where we have a strictly vertical or horizontal movement along edges
  if (absDeltaX == 0  ||  absDeltaY == 0)
  {
    uint8_t diagDirX2, diagDirY2;

    if (absDeltaX == 0)
    {
      // Handle edge case where start position is on the rightmost edge of the board,
      // so you must move left for the first diagonal motion. Move right for first
      // diagonal in all other cases of strictly vertical motion.
      diagDirX = (startCol == (TOTAL_UNITSPACES - 1)) ? -1 : 1;

      subDeltaX = 0;
      diagDirX2 = -diagDirX;
    }
    else
    {
      // Handle edge case where start position is on the topmost edge of the board,
      // so you must move down for the first diagonal motion. Move up for first
      // diagonal in all other cases of strictly horizontal motion.
      diagDirY = (startRow == (TOTAL_UNITSPACES - 1)) ? -1 : 1;

      subDeltaY = 0; 
      diagDirY2 = -diagDirY;
    }

    // Add diagonal movement
    cols[1] = cols[0] + diagDirX;
    rows[1] = rows[0] + diagDirY;

    // Add straight movement
    cols[2] = cols[1] + subDeltaX;
    rows[2] = rows[1] + subDeltaY;

    // Add diagonal movement, with flipped direction
    cols[3] = cols[2] + diagDirX2;
    rows[3] = rows[2] + diagDirY2;

    // 3 added points, plus the initial point
    numPoints = 4;
  }
  // Case where we're moving a cached piece to the graveyard from a capture
  else if (startCol % 2  ||  startRow % 2)
  {
    subDeltaX = deltaX - diagDirX; 
    subDeltaY = deltaY - diagDirY;

    // Note: at least one of subDeltaX or subDeltaY is equal to 0
    // Add straight X movement
    cols[1] = cols[0] + subDeltaX;
    rows[1] = rows[0];

    // Add straight Y movement
    cols[2] = cols[1];
    rows[2] = rows[1] + subDeltaY;

    // Add diagonal movement
    cols[3] = cols[2] + diagDirX;
    rows[3] = rows[2] + diagDirY;

    // 3 added points, plus the initial point
    numPoints = 4;
  }
  // Case where we have a knight or graveyard movement
  else if (absDeltaX == 2  ||  absDeltaY == 2)
  {
    if (absDeltaX == 2)
      subDeltaX = 0;
    else
      subDeltaY = 0;

    // Add diagonal movement
    cols[1] = cols[0] + diagDirX;
    rows[1] = rows[0] + diagDirY;

    // Add straight Y movement
    cols[2] = cols[1] + subDeltaX;
    rows[2] = rows[1] + subDeltaY;

    // Add diagonal movement
    cols[3] = cols[2] + diagDirX;
    rows[3] = rows[2] + diagDirY;

    // 3 added points, plus the initial point
    numPoints = 4;
  }
  // Case where we have all of the 4 sub-movements mentioned before the function signature
  else
  {
    // Add diagonal movement
    cols[1] = cols[0] + diagDirX;
    rows[1] = rows[0] + diagDirY;

    // Note: at least one of subDeltaX or subDeltaY is equal to 0
    // Add straight X movement 
    cols[2] = cols[1] + subDeltaX;
    rows[2] = rows[1];

    // Add straight Y movement
    cols[3] = cols[2];
    rows[3] = rows[2] + subDeltaY;

    // Add diagonal movement
    cols[4] = cols[3] + diagDirX;
    rows[4] = rows[3] + diagDirY;

    // 4 added points, plus the initial point
    numPoints = 5;
  }
  

  // Enable electromagnet
  ledcWrite(EM_PWM_CHANNEL, PWM_HALF);
  
  // Loop through each of the calculated points and call the according movement function
  // Start from 1 since 0 is the start point and we always refer back to it
  pointCounter = 1;
  while (pointCounter < numPoints)
  {
    // Note: rows[0] and cols[0] collectively store the start point requested by the function
    // That allows us to modularly make the calls in a nice for-loop  
    if (rows[pointCounter] == rows[pointCounter-1])
    {
      statusCodeResult = moveStraight(yMotor, cols[pointCounter], rows[pointCounter]);
      if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        break;
    }
    else if (cols[pointCounter] == cols[pointCounter-1])
    {
      statusCodeResult = moveStraight(xMotor, cols[pointCounter], rows[pointCounter]);
      if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        break;
    }
    else
    {
      statusCodeResult = moveDiagonal(cols[pointCounter], rows[pointCounter]);
      if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        break;
    }

    pointCounter++;
  }

  // Turn electromagnet off
  digitalWrite(ELECTROMAGNET, LOW);

  return (pointCounter == numPoints);
}

bool alignPiece(int col, int row)
{
  return true;
}

// Assumes that the starting position is the center of the square the piece is on
void centerPiece()
{
  // Loop counter
  int i;

  moveToFirstCircle();

  // Enable electromagnet
  ledcWrite(EM_PWM_CHANNEL, PWM_HALF);

  for (i = 0; i < NUM_CIRCLES; i++)
  {
    // Both functions are called with currentCircle, quarter
    // Circle 0 starts at the top, circle 1 at the left, circle 2 at the bottom, 
    // continuing in a counterclockwise fashion. i % 4 is used in case there
    // are more than four circles. Circle 4 should start at the top, 5 at the left, etc
    makeCircle(i, i % 4);
    moveToNextCircle(i, i % 4);
  }

  // Turn electromagnet off
  digitalWrite(ELECTROMAGNET, LOW);
}

// Handles error codes passed in by the returns of relevant movement functions
// Must take in an error code to work properly, returns true by default
bool statusCodeHandler(uint8_t status)
{
  if (status == HIT_POS_X_ENDSTOP)
  {
    alignAxis(xMotor, MAX_POSITION);
  }
  else if (status == HIT_POS_Y_ENDSTOP)
  {
    alignAxis(yMotor, MAX_POSITION);
  }
  else if (status == HIT_NEG_X_ENDSTOP)
  {
    alignAxis(xMotor, ZERO_POSITION);
  }
  else if (status == HIT_NEG_Y_ENDSTOP)
  {
    alignAxis(yMotor, ZERO_POSITION);
  }
  else if (status == INVALID_ALIGNMENT)
  {
    home();
  }
  else
  {
    // Returns false if status code not one of those handled above, 
    // e.g. INVALID_ARGS, or if status code is invalid
    return false;
  }
  
  return true;
}
