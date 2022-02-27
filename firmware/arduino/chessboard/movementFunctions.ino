bool moveDirect(int startCol, int startRow, int endCol, int endRow)
{
    return true;
}

bool moveAlongEdges(int startCol, int startRow, int endCol, int endRow)
{
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
