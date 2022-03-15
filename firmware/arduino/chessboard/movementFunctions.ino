bool moveDirect(int startCol, int startRow, int endCol, int endRow)
{
    return true;
}

bool moveAlongEdges(int startCol, int startRow, int endCol, int endRow)
{
    // There are 4 points total, but only 3 total for each x/y component
    uint8_t rows[3], cols[3];
    uint8_t dirX, dirY;
    uint8_t statusCodeResult;

    // Make initial move to first position. Move diagonally since it's faster.
    statusCodeResult = moveDiagonal(startCol, startRow);
    if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        return false;

    // This type of move will always have a set of 4 moves:
    // ====================================================
    // 1. Move to edge from center of square
    // 2. Move along X axis
    // 3. Move along Y axis
    // 4. Move to center of square from edge

    // Calculate directions for x and y
    dirX = (endCol > startCol) ? POS_DIR : NEG_DIR;
    dirY = (endRow > startRow) ? POS_DIR : NEG_DIR;

    // Calculate the points for each of the 4 moves listed above
    // We use previously calculated points to calculate the next points, since they're on the same path
    // Move 1
    cols[0] = startCol + (dirX == POS_DIR) ? 1 : -1;
    rows[0] = startRow + (dirY == POS_DIR) ? 1 : -1;

    // Move 2
    // Subtract two to account for moves 1 and 4 both moving one unitspace
    cols[1] = cols[0] + (endCol - startCol - 2);

    // Move 3
    // Subtract two to account for moves 1 and 4 both moving one unitspace
    rows[1] = rows[0] + (endRow - startRow - 2);

    // Move 4
    cols[2] = cols[1] + (dirX == POS_DIR) ? 1 : -1;
    rows[2] = rows[1] + (dirY == POS_DIR) ? 1 : -1;

    
    // Use the calculated points and call the according movement functions
    // Move 1
    statusCodeResult = moveDiagonal(cols[0], rows[0]);
    if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        return false;

    // Move 2
    statusCodeResult = moveStraight(xMotor, cols[1], rows[0]);
    if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        return false;

    // Move 3
    statusCodeResult = moveStraight(yMotor, cols[1], rows[1]);
    if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        return false;

    // Move 3
    statusCodeResult = moveDiagonal(cols[2], rows[2]);
    if (statusCodeResult != SUCCESS && !statusCodeHandler(statusCodeResult))
        return false;

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
