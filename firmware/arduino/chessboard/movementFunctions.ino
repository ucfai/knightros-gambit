bool moveDirect(int startCol, int startRow, int endCol, int endRow)
{
  // TODO:
  // Must account for diagonal and straight movement cases
  // Must also return true if the current position is the target position 
  // (Since no change is needed)
  return true;
}

bool moveAlongEdges(int startCol, int startRow, int endCol, int endRow)
{
  // There are 4 points total, but only 3 total for each x/y component
  uint8_t rows[5], cols[5];
  int8_t dirX, dirY;
  uint8_t i, statusCodeResult;
  uint8_t curCol, curRow;

  curCol = currPositionX / (stepsPerUnitSpace * 8);
  curRow = currPositionY / (stepsPerUnitSpace * 8);

  // If target point is equal to the start point
  if (endCol == startCol  &&  endRow == startRow)
    return true;
  
  // Make initial move to first position. Move diagonally since it's faster.
  if (!moveDirect(curCol, curRow, startCol, startRow))
    return false;

  // This type of move will always have a set of 4 moves:
  // ====================================================
  // 1. Move to edge from center of square
  // 2. Move along X axis (optional)
  // 3. Move along Y axis (optional)
  // 4. Move to center of square from edge
  // Steps 2 and 3 are considered optional because those moves may not exist

  // Calculate directions for x and y
  dirX = (endCol > startCol) ? 1 : -1;
  dirY = (endRow > startRow) ? 1 : -1;

  // Calculate the points for each of the 4 moves listed above
  // We use previous points to calculate subsequent points, since they're on the same path
  // Initial position
  cols[0] = startCol;
  rows[0] = startRow; 
  
  // Move 1
  cols[1] = cols[0] + dirX;
  rows[1] = rows[0] + dirY;

  // Move 2
  // Sutract two to account for moves 1 and 4 both moving one unitspace
  cols[2] = cols[1] + (endCol - startCol - 2);
  rows[2] = rows[1];

  // Move 3
  // Subtract two to account for moves 1 and 4 both moving one unitspace
  rows[3] = rows[2] + (endRow - startRow - 2);
  cols[3] = cols[2];

  // Move 4
  cols[4] = cols[3] + dirX;
  rows[4] = rows[3] + dirY;

  
  // Use the calculated points and call the according movement functions

  // TODO:
  // Remove magic number here
  for (i = 1; i < 5; i++)
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
