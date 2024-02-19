import java.io.*;
import java.net.*;
public class FileServer
{
    public static void main(String[] args)
    {
        new FileServer();
    }   
    public FileServer()
    {
        DataOutputStream output;
        DataInputStream input;
        Socket socket;
        ServerSocket serversocket;
        BufferedReader br;
        String everything;
        try
        {
            serversocket=new ServerSocket(8000);
            System.out.println("Server Started.......");
            socket=serversocket.accept();
            input=new DataInputStream(socket.getInputStream());
            output=new DataOutputStream(socket.getOutputStream());
            while(true)
            {
                String str =input.readUTF();
                System.out.println(str);
                try{
                    URL url = getClass().getResource(str);
                    InputStream istream= url.openStream();

                    br=new BufferedReader(new InputStreamReader(istream));
                    StringBuilder sb=new StringBuilder();
                    String line = br.readLine();
                    while(line!=null)
                    {
                        sb.append(line);
                        line=br.readLine();
                    }
                    everything=sb.toString();
                }
                catch(Exception ex)
                {
                    everything = "File Not Found";
                }
                output.writeUTF(everything);
            }
        }
        catch(Exception ex)
        {
            everything="Error";
        }
        finally{
        }
    }
}