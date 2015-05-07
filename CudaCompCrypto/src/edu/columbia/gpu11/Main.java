package edu.columbia.gpu11;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.HashMap;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

public class Main extends JFrame implements ActionListener, DocumentListener {

	private static final long serialVersionUID = 156323633057513646L;

	private String method = "";

	private JLabel methodLbl, inputLbl, outputLbl, passLbl, publicLbl,
			privateLbl;
	private JTextField inputText, outputText, passText, publicText,
			privateText;

	private JButton doBtn, undoBtn, genBtn, inputBtn, outputBtn, publicBtn,
			privateBtn;
	private JComboBox<String> methodCmb;

	private JFileChooser chooser;
	private CuWrapper wrapper;

	private HashMap<Integer, String> errMap;

	/**
	 * Start the program and load native code
	 * @param args
	 */
	public static void main(String[] args) {
		System.loadLibrary("SharedCryptography");

		Main main = new Main();
		main.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		main.setVisible(true);
		main.setResizable(false);
	}

	/**
	 * Initialize the user interface
	 */
	public Main() {
		setLocationRelativeTo(null);
		setTitle("Cuda-Comp-Crypto");

		setSize(550, 340);
		setLayout(null);
		
		methodLbl = new JLabel("Method:");
		methodLbl.setBounds(10, 10, 80, 30);
		add(methodLbl);

		methodCmb = new JComboBox<String>();
		methodCmb.setBounds(100, 10, 90, 30);
		add(methodCmb);

		methodCmb.addItem("");

		methodCmb.addItem("DES (CPU)");
		methodCmb.addItem("DES (GPU)");

		methodCmb.addItem("AES (CPU)");
		methodCmb.addItem("AES (GPU)");

		methodCmb.addItem("RSA (CPU)");
		methodCmb.addItem("RSA (GPU)");

		methodCmb.addActionListener(this);

		inputLbl = new JLabel("Input File:");
		inputLbl.setBounds(10, 60, 80, 30);
		add(inputLbl);

		inputText = new JTextField();
		inputText.setBounds(100, 60, 340, 30);
		add(inputText);

		inputText.getDocument().addDocumentListener(this);

		outputLbl = new JLabel("Output File:");
		outputLbl.setBounds(10, 110, 80, 30);
		add(outputLbl);

		outputText = new JTextField();
		outputText.setBounds(100, 110, 340, 30);
		add(outputText);

		outputText.getDocument().addDocumentListener(this);

		passLbl = new JLabel("Password:");
		passLbl.setBounds(10, 160, 80, 30);
		add(passLbl);

		passText = new JTextField();
		passText.setBounds(100, 160, 440, 30);
		add(passText);

		passText.getDocument().addDocumentListener(this);

		publicLbl = new JLabel("Public Key:");
		publicLbl.setBounds(10, 210, 80, 30);
		add(publicLbl);

		publicText = new JTextField();
		publicText.setBounds(100, 210, 340, 30);
		add(publicText);

		publicText.getDocument().addDocumentListener(this);

		privateLbl = new JLabel("Private Key:");
		privateLbl.setBounds(10, 260, 80, 30);
		add(privateLbl);

		privateText = new JTextField();
		privateText.setBounds(100, 260, 340, 30);
		add(privateText);

		privateText.getDocument().addDocumentListener(this);

		doBtn = new JButton("Encrpyt");
		doBtn.setBounds(350, 10, 90, 30);
		add(doBtn);

		doBtn.addActionListener(this);

		undoBtn = new JButton("Decrpyt");
		undoBtn.setBounds(450, 10, 90, 30);
		add(undoBtn);

		undoBtn.addActionListener(this);

		genBtn = new JButton("Generate Keys");
		genBtn.setBounds(200, 10, 140, 30);
		add(genBtn);

		genBtn.addActionListener(this);

		inputBtn = new JButton("Browse");
		inputBtn.setBounds(450, 60, 90, 30);
		add(inputBtn);

		inputBtn.addActionListener(this);

		outputBtn = new JButton("Browse");
		outputBtn.setBounds(450, 110, 90, 30);
		add(outputBtn);

		outputBtn.addActionListener(this);

		publicBtn = new JButton("Browse");
		publicBtn.setBounds(450, 210, 90, 30);
		add(publicBtn);

		publicBtn.addActionListener(this);

		privateBtn = new JButton("Browse");
		privateBtn.setBounds(450, 260, 90, 30);
		add(privateBtn);

		privateBtn.addActionListener(this);

		methodCmb.setSelectedIndex(0);

		chooser = new JFileChooser();
		chooser.setAcceptAllFileFilterUsed(false);
		chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
		chooser.setFileHidingEnabled(false);

		wrapper = new CuWrapper();

		// Map the error code and the textual messages
		
		errMap = new HashMap<Integer, String>();
		errMap.put(-1, "Key Error");
		errMap.put(-2, "Text Error");

		errMap.put(-3, "Input File Error");
		errMap.put(-4, "Output File Error");

		errMap.put(-5, "GPU Device Error");

		errMap.put(-6, "Wrong Algorithm");
		errMap.put(-7, "Wrong Device");
	}

	/**
	 * Switch the cyptographic algorithm
	 * @param newMethod
	 */
	protected void switchMethod(String newMethod) {
		method = newMethod;

		if (method.startsWith("DES") || method.startsWith("AES")) {
			inputLbl.setEnabled(true);
			outputLbl.setEnabled(true);
			passLbl.setEnabled(true);
			publicLbl.setEnabled(false);
			privateLbl.setEnabled(false);

			inputText.setEnabled(true);
			outputText.setEnabled(true);
			passText.setEnabled(true);
			publicText.setEnabled(false);
			privateText.setEnabled(false);

			inputBtn.setEnabled(true);
			outputBtn.setEnabled(true);

			publicBtn.setEnabled(false);
			privateBtn.setEnabled(false);

		} else if (method.startsWith("RSA")) {
			inputLbl.setEnabled(true);
			outputLbl.setEnabled(true);
			passLbl.setEnabled(false);
			publicLbl.setEnabled(true);
			privateLbl.setEnabled(true);

			inputText.setEnabled(true);
			outputText.setEnabled(true);
			passText.setEnabled(false);
			publicText.setEnabled(true);
			privateText.setEnabled(true);

			inputBtn.setEnabled(true);
			outputBtn.setEnabled(true);

			publicBtn.setEnabled(true);
			privateBtn.setEnabled(true);

		} else {
			inputLbl.setEnabled(false);
			outputLbl.setEnabled(false);
			passLbl.setEnabled(false);
			publicLbl.setEnabled(false);
			privateLbl.setEnabled(false);

			inputText.setEnabled(false);
			outputText.setEnabled(false);
			passText.setEnabled(false);
			publicText.setEnabled(false);
			privateText.setEnabled(false);

			inputBtn.setEnabled(false);
			outputBtn.setEnabled(false);

			publicBtn.setEnabled(false);
			privateBtn.setEnabled(false);
		}

		doBtn.setEnabled(canDo());
		undoBtn.setEnabled(canUndo());
		genBtn.setEnabled(canGenerateKey());
	}

	private boolean canDo() {
		if (inputText.getText().trim().isEmpty()
				|| outputText.getText().trim().isEmpty()) {
			return false;
		}

		int passLen = passText.getText().length();

		if (method.startsWith("DES")) {
			return passLen == 8;
		} else if (method.startsWith("AES")) {
			return passLen == 16 || passLen == 24 || passLen == 32;
		} else if (method.startsWith("RSA")) {
			return method.startsWith("RSA")
					&& !publicText.getText().trim().isEmpty();
		}

		return false;
	}

	private boolean canUndo() {
		if (inputText.getText().trim().isEmpty()
				|| outputText.getText().trim().isEmpty()) {
			return false;
		}

		int passLen = passText.getText().length();

		if (method.startsWith("DES")) {
			return passLen == 8;
		} else if (method.startsWith("AES")) {
			return passLen == 16 || passLen == 24 || passLen == 32;
		} else if (method.startsWith("RSA")) {
			return method.startsWith("RSA")
					&& !privateText.getText().trim().isEmpty();
		}

		return false;
	}

	private boolean canGenerateKey() {
		return method.startsWith("RSA")
				&& !publicText.getText().trim().isEmpty()
				&& !privateText.getText().trim().isEmpty();
	}

	/**
	 * Click buttons or change combo box
	 */
	@Override
	public void actionPerformed(ActionEvent arg0) {
		if (arg0.getSource() == methodCmb) {
			switchMethod(methodCmb.getSelectedItem().toString());
			return;
		} else if (arg0.getSource() == inputBtn) {
			chooser.showOpenDialog(this);

			File file = chooser.getSelectedFile();
			if (file == null) {
				return;
			}

			inputText.setText(file.getAbsolutePath());
		} else if (arg0.getSource() == outputBtn) {
			chooser.showSaveDialog(this);

			File file = chooser.getSelectedFile();
			if (file == null) {
				return;
			}

			outputText.setText(file.getAbsolutePath());
		} else if (arg0.getSource() == publicBtn) {
			chooser.showDialog(this, "Set Public Key");

			File file = chooser.getSelectedFile();
			if (file == null) {
				return;
			}

			publicText.setText(file.getAbsolutePath());
		} else if (arg0.getSource() == privateBtn) {
			chooser.showDialog(this, "Set Private Key");

			File file = chooser.getSelectedFile();
			if (file == null) {
				return;
			}

			privateText.setText(file.getAbsolutePath());
		} else if (arg0.getSource() == doBtn) {
			int idx = methodCmb.getSelectedIndex() - 1;

			int alg = idx >> 1;
			int dev = idx & 1;

			String inName = inputText.getText().trim();
			String outName = outputText.getText().trim();

			String arg4 = null;

			if (method.startsWith("DES") || method.startsWith("AES")) {
				arg4 = passText.getText().trim();
			} else if (method.startsWith("RSA")) {
				arg4 = publicText.getText().trim();
			}

			float result = wrapper
					.doAlgo(alg, dev, inName, outName, arg4);
			showResult(result);
		} else if (arg0.getSource() == undoBtn) {
			int idx = methodCmb.getSelectedIndex() - 1;

			int alg = idx >> 1;
			int dev = idx & 1;

			String inName = inputText.getText().trim();
			String outName = outputText.getText().trim();

			String arg4 = null;

			if (method.startsWith("DES") || method.startsWith("AES")) {
				arg4 = passText.getText();
			} else if (method.startsWith("RSA")) {
				arg4 = privateText.getText().trim();
			}

			float result = wrapper.undoAlgo(alg, dev, inName, outName, arg4);
			showResult(result);
		} else if (arg0.getSource() == genBtn) {
			int idx = methodCmb.getSelectedIndex() - 1;

			int dev = idx & 1;
			
			String publicKey = publicText.getText().trim();
			String privateKey = privateText.getText().trim();

			float result = wrapper.genRSA(dev, publicKey, privateKey);
			showResult(result);
		}
	}

	private void showResult(float result) {
		if (result > -1e-6) {
			JOptionPane.showMessageDialog(this, "Completed in " + result
					+ " milliseconds", "Operation Success",
					JOptionPane.INFORMATION_MESSAGE);
		} else {
			int errCode = (int) result;

			if (errMap.containsKey(errCode)) {
				String errMsg = errMap.get(errCode);

				JOptionPane.showMessageDialog(this, errMsg + "!",
						"Operation Failed", JOptionPane.ERROR_MESSAGE);
			}
		}
	}

	@Override
	public void changedUpdate(DocumentEvent arg0) {
		doBtn.setEnabled(canDo());
		undoBtn.setEnabled(canUndo());
		genBtn.setEnabled(canGenerateKey());
	}

	@Override
	public void insertUpdate(DocumentEvent arg0) {
		doBtn.setEnabled(canDo());
		undoBtn.setEnabled(canUndo());
		genBtn.setEnabled(canGenerateKey());
	}

	@Override
	public void removeUpdate(DocumentEvent arg0) {
		doBtn.setEnabled(canDo());
		undoBtn.setEnabled(canUndo());
		genBtn.setEnabled(canGenerateKey());
	}
}
