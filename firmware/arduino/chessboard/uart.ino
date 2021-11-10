char incomingByte;
char  buffer[5];
int byteNum = -1; //-1 indicates that the start code hasn't been received

//Wait for input
void serialEvent2()
{
    //Loop through all available bytes
    while (Serial2.available())
    {
        //Get byte
        incomingByte = (char) Serial2.read();

        //Reset buffer position
        if (incomingByte == '~')
        {
            byteNum = 0;
        }
        //Add byte to buffer
        else if (byteNum != -1)
        {
            buffer[byteNum] = incomingByte;
        }

        //Check if the buffer is full, process input
        if (byteNum == 4)
        {
            //Reset buffer position
            byteNum = -1;

            //Process input
            switch(buffer[0])
            {
                case 0:
                    //move_straight(int startRow, int startCol, int endRow, int endCol)
                    move_straight(buffer[1] - 'a', buffer[2] - '0', buffer[3] - 'a', buffer[4] - '0');
                    break;
                case 1:
                    //move_to_graveyard(int startRow, int startCol, char color, char type)
                    move_to_graveyard(buffer[1] - 'a', buffer[2] - '0', buffer[3], buffer[4]);
                    break;
                case 2:
                    //Args: piece color (b/w), kingside or queenside (-/0) 
                    //castle(char color, char side)
                    castle(buffer[1], buffer[3]);
                    break;
                case 3:
                    //move_along_edges(int startRow, int startCol, int endRow, int endCol)
                    move_along_edges(buffer[1] - 'a', buffer[2] - '0', buffer[3] - 'a', buffer[4] - '0');
                    break;
                default:
                    //Handle error. Idk how we plan on doing this yet.
                    break;
            }
        }
        //Increment buffer position
        else if (byteNum != -1)
        {
            byteNum++;
        }
    }
}