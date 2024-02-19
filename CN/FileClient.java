import java.io.*;
import java.net.*;
public class FileClient
{
    public static void main(String[] args)
    {
        new FileClient();
    }
    public FileClient(){
        BufferedReader bufReader=new BufferedReader(new InputStreamReader(System.in));
        try{
            Socket clientsocket=new Socket("localhost",8000);
            System.out.println("Connecting to server...");
            DataInputStream input=new DataInputStream(clientsocket.getInputStream());
            DataOutputStream output = new DataOutputStream(clientsocket.getOutputStream());
            System.out.println("Enter file name:");
            String Name = bufReader.readLine();
            output.writeUTF(Name);
            String EcFile = input.readUTF();
            System.out.println(EcFile);
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
