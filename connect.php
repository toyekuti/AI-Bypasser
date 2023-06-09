<?php
	$fullname = $_POST['fullname'];
	$plan = $_POST['plan'];
	$email = $_POST['email'];
	$pass = $_POST['pass'];
	

	$host = "localhost";
	$dbname = "Datab";
	$username = "root";
	$password = "Alisson1911&rox";

	$conn = mysqli_connect($host,$username,$password,$dbname);

	if(mysqli_connect_errno()){
		die("Connection error: ".mysqli_connect_error());
	}
	
	$sql = "INSERT INTO people (fullname,plan,email,pass) VALUES (?, ?, ?, ?)";

	$stmt = mysqli_stmt_init($conn);
	if(! mysqli_stmt_prepare($stmt,$sql)){
		die(mysqli_error($conn));
	}
	mysqli_stmt_bind_param($stmt,"ssss",$fullname,$plan,$email,$pass);
	mysqli_stmt_execute($stmt);
	header("Location: StepForm\multistep.html");


	// $conn = new mysqli('localhost','root','Alisson1911&rox','datab');
	// if($conn->connect_error){
	// 	die('Connection Failed : '.$conn->connect_error);;
	// }else{
	// 	$stmt = $conn->prepare("insert into people(fullname,plan,email,pass) values(?,?,?,?)");
	// 	$stmt->bind_param("ssss",$fullname,$plan,$email,$pass);
	// 	$stmt->execute();
	// 	echo "Successfull Registration";
	// 	$stmt->close();
	// 	$conn->close();
	// }
	// print_r($_POST);

?>