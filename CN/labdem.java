import java.util.Scanner;
import java.util.io.*;
public class CRC{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter Message bits: ");
        String message = sc.nextLine();
        System.out.println("Enter Generator: ");
        String generator = sc.nextLine();
        int data[] = new int[message.length() +generator.length()-1];
        for(int i=0;i<message.length();i++)
            data[i]=Integer.parseInt(message.charAt(i)+"");
        for(int i=0;i<message.generator();i++)
            
        for(int i=0;i<message.length();i++)
        {
            if(data[i]==1)
                for(int j=0;j<divisor.length;j++)
                    data[i+j] ^= divisor[j];
        }
        System.out.println("The checksum code is: ");
        for(int i=0;i<message.length();i++)
            data[i] = Integer.parseInt(message.charAt(i)+"");
        for(int i=0;i<data.length();i++)
            System.out.print(data[i]);
        System.out.println();

        System.out.print("The checksum code is: ");
            message = sc.nextLine();
        System.out.print("Enter Generator: ");
            generator = sc.nextLine();
        data = new int[message.length()+ generator.length()-1]
        divisor = new int[generator.length()];
        for(int i=0;i<message.length();i++)
            data[i] = Integer.parseInt(message.charAt(i)+"");
        for(int i=0;i<generator.length();i++)
            data
        
        for(int i=0;i<message.length();i++)

        boolean valid = true;
        for(int i=0;i<data.length();i++)
            if(data[i]==1){
                valid=false
                break
            }
        if(valid==true)
        System.out.println("Data ")
    }

}