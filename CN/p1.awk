BEGIN{c=0;

       r=0;
}

{
        if($1=="d")
        {
               c++;
         }
         else if($1=="r")
         {
		r++;
	}
}
END{ printf("The number of packets dropped=%d\n",c);
   printf("The number of packets recieved =%d\n",r); }
