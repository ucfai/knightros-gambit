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
    analogWrite(ELECTROMAGNET, PWM_SCALE);

    for (i = 0; i < NUM_CIRCLES; i++)
    {
        makeCircle(i);
        moveToNextCircle(i);
    }

    // Turn electromagnet off
    digitalWrite(ELECTROMAGNET, LOW);
}