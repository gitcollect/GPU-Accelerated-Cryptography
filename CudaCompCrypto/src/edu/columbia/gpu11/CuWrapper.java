package edu.columbia.gpu11;

/**
 * Call native cryptographic algorithms
 * @author Yuqing Guan
 *
 */
public class CuWrapper {
	private native void init();
	public native float doAlgo(int alg, int dev, String inName, String outName, String arg4);
	public native float undoAlgo(int alg, int dev, String inName, String outName, String arg4);
	public native float genRSA(int dev, String publicKey, String privateKey);
	
	public CuWrapper()
	{
		init();
	}
}
