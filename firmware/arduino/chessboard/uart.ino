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
            buffer[byteNum++] = incomingByte;
        }

        //Check if the buffer is full, process input
        if (byteNum == 5)
        {
            //Reset buffer position
            byteNum = -1;

            //Process input
            //Returns true for valid input and false for invalid input. Calls movement function
            parseMessageFromPi(buffer);
        }
    }
}

bool parseMessageFromPi(char * message)
{
    //Move types 0 and 3 are valid for the same range of values
    if (message[0] == 0 || message[0] == 3)
    {
        //Check that the inputs are in a valid range
        if (message[1] < 'a' || message[1] > 'h' || message[2] < '1' ||  message[2] > '8' || message[3] < 'a' || message[3] > 'h' || message[4] < '1' ||  message[4] > '8') 
        {
            return false;
        }

        //Move type 0
        if (message[0] == 0)
            moveStraight(message[1] - 'a', message[2] - '1', message[3] - 'a', message[4] - '1');
        //Move type 3
        else
            moveAlongEdges(message[1] - 'a', message[2] - '1', message[3] - 'a', message[4] - '1');

    }
    else if (message[0] == 1)
    {
        //Check validity for the first 3 bytes
        if (message[1] < 'a' || message[1] > 'h' || message[2] < '1' ||  message[2] > '8' || ((message[3] != 'b') && message[3] != 'w'))
            return false;
        
        //Check for valid piece type
        if (!(message[4] == 'p' || message[4] == 'r' || message[4] == 'k' || message[4] == 'b' || message[4] == 'q'))
            return false;

        //Move type 1
        moveToGraveyard(message[1] - 'a', message[2] - '1', message[3], message[4]);        
    }
    else if (message[0] == 2)
    {
        //Check input validity
        if (!(message[1] == 'b' || message[1] == 'w') || message[2] != '0' || message[4] != '0' || !(message[3] == '0' || message[3] == '-'))
            return false;

        //Move type 2
        castle(message[1], message[3]);
    }
    else
        return false;

    //Move is valid and was made
    return true;
}