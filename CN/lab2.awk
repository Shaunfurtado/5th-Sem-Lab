BEGIN{
	count=0;
}
{
	if($1=="d")
	{
		count++;
	}
}
END{ printf("The number of packets dropped =%d\n",count);
}

