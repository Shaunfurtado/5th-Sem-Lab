import java.util.Scanner;

public class CRCb {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter message bits:");
        String message = sc.nextLine();
        System.out.println("Enter generator (16 bits):");
        String generator = sc.nextLine();

        // Convert message and generator strings to arrays of integers
        int[] data = new int[message.length() + generator.length() - 1];
        int[] divisor = new int[generator.length()];
        for (int i = 0; i < message.length(); i++)
            data[i] = Integer.parseInt(message.charAt(i) + "");
        for (int i = 0; i < generator.length(); i++)
            divisor[i] = Integer.parseInt(generator.charAt(i) + "");

        // Perform CRC division
        for (int i = 0; i < message.length(); i++) {
            if (data[i] == 1) {
                for (int j = 0; j < divisor.length; j++)
                    data[i + j] ^= divisor[j];
            }
        }

        // Generate checksum code
        System.out.println("The checksum code is:");
        for (int i = 0; i < data.length; i++)
            System.out.print(data[i]);
        System.out.println();

        // Check validity of data stream
        System.out.println("Enter received data bits:");
        String receivedData = sc.nextLine();
        data = new int[receivedData.length() + generator.length() - 1];
        for (int i = 0; i < receivedData.length(); i++)
            data[i] = Integer.parseInt(receivedData.charAt(i) + "");

        // Perform CRC division on received data
        for (int i = 0; i < receivedData.length(); i++) {
            if (data[i] == 1) {
                for (int j = 0; j < divisor.length; j++)
                    data[i + j] ^= divisor[j];
            }
        }
        boolean valid = true;
        for (int i = 0; i < data.length; i++) {
            if (data[i] == 1) {
                valid = false;
                break;
            }
        }
        if (valid)
            System.out.println("Data stream is valid.");
        else
            System.out.println("Data stream is invalid. CRC error has occurred.");
    }
}
