bool moveDirect(int startCol, int startRow, int endCol, int endRow)
{
  // TODO:
  // Must account for diagonal and straight movement cases
  // Must also return true if the current position is the target position 
  // (Since no change is needed)
  return true;
}

// This type of move is typically decomposed into a set of 4 moves:
// ================================================================
// 1. Move to edge from center of square
// 2. Move along X axis (optional)
// 3. Move along Y axis (optional)
// 4. Move to center of square from edge

// Some of these moves may not happen, which is accounted for below
bool moveAlongEdges(int startCol, int startRow, int endCol, int endRow)
{
  // There are 4 points total, but only 3 total for each x/y component
  uint8_t rows[5], cols[5];
  int8_t diagDirX, diagDirY;
  int8_t deltaX, deltaY, subDeltaX, subDeltaY, absDeltaX, absDeltaY;
  uint8_t i, statusCodeResult;
  uint8_t curCol, curRow;
  uint8_t numPoints = 0;

  curCol = currPositionX / (stepsPerUnitSpace * 8);
  curRow = currPositionY / (stepsPerUnitSpace * 8);

  // Find the signed difference between the finial and initial points
  deltaX = endCol - startCol;
  deltaY = endRow - startRow;

  // We only need the magnitude of deltaX and deltaY after this point 
  absDeltaX = abs(deltaX);
  absDeltaY = abs(deltaY);

  // Calculate the direction of the diagonals
  diagDirX = (deltaX > 0) ? 1 : -1;
  diagDirY = (deltaY > 0) ? 1 : -1;

  // If target point is equal to the start point
  if (deltaX == 0  &&  deltaY == 0)
    return true;
  
  // Make initial move to first position. Move diagonally since it's faster.
  if (!moveDirect(curCol, curRow, startCol, startRow))
    return false;

  // Calculate a list of up to 4 new points (including the start point) 
  // We use previous points to calculate subsequent points, since they're on the same path
  // Initial position
  cols[0] = startCol;
  rows[0] = startRow;

  // Case where we should call moveDirect, since pathing can be simplified to that
  if (absDeltaX <= 2  &&  absDeltaY <= 2)
    return moveDirect(startCol, startRow, endCol, endRow);

  // Because of the checks above, we know absDeltaX and absDeltaY can't both be 0 at the same time
  // Case where we have a strictly vertical or horizontal movement along edges
  if (absDeltaX == 0  ||  absDeltaY == 0)
  {
    uint8_t diagDirX2, diagDirY2;

    if (absDeltaX == 0)
    {
      // Handle edge case where start position is on the edge of the board, so you must move inward
      dirX = (startCol == (TOTAL_UNITSPACES - 1)) ? -1 : 1;

      subDeltaX = 0;
      subDeltaY = deltaY - (2 * diagDirY);  
      diagDirX2 = -diagDirX;
    }
    else
    {
      // Handle edge case where start position is on the edge of the board, so you must move inward
      dirY = (startRow == (TOTAL_UNITSPACES - 1)) ? -1 : 1;

      subDeltaX = deltaX - (2 * diagDirX);
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

  /*
  // Calculate directions for x and y
  diagDirX = (deltaX > 0) ? 1 : -1;
  diagDirY = (deltaY > 0) ? 1 : -1;

  subDeltaX = deltaX - (2 * diagDirX);
  subDeltaY = deltaY - (2 * diagDirY);
  */

  
  // Use the calculated points and call the according movement functions

  // TODO:
  // Remove magic number here
  for (i = 1; i < numPoints; i++)
  {
    // TODO:
    // This is where we can account for edge cases, by testing distance for example
    // We should also make checks for when i == 2 or i == 3 specifically, since that's where we'll
    // encounter either the horizontal or vertical travel being 0

    // Note: rows[0] and cols[0] collectively store the start point requested by the function
    // That allows us to modularly make the calls in a nice for-loop  
    if (rows[i] == rows[i-1])
      moveStraight(yMotor, cols[i], rows[i]);
    else if (cols[i] == cols[i-1])
      moveStraight(xMotor, cols[i], rows[i]);
    else
      moveDiagonal(cols[i], rows[i]);
  }

  return true;
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
  // Returns false if status code not one of those handled above, e.g. INVALID_ARGS, or if status code is invalid
  else
  {
    return false;
  }
  
  return true;
}
